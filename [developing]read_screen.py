import cv2
import numpy as np
import subprocess
import time
import queue
import threading
import sys
from typing import Optional, Tuple

# ==================== 配置区域 ====================
SCRCPY_PATH = r"C:\scrcpy\scrcpy.exe"
ADB_PATH = r"C:\scrcpy\adb.exe"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
DEFAULT_RESOLUTION = (1080, 2400)  # 默认分辨率
MAX_RETRIES = 3  # 最大重试次数
BUFFERED_FRAMES = 5  # 帧缓冲数量


# ==================== 工具函数 ====================
def run_adb_command(cmd: list) -> bool:
    """执行ADB命令并检查结果"""
    try:
        subprocess.run([ADB_PATH] + cmd, check=True, timeout=10)
        return True
    except subprocess.SubprocessError as e:
        print(f"ADB命令失败: {' '.join(cmd)} | 错误: {str(e)}")
        return False


def reset_device_connection():
    """重置设备连接"""
    print("🔄 重置ADB连接...")
    run_adb_command(["kill-server"])
    run_adb_command(["start-server"])
    if not run_adb_command(["devices"]):
        raise RuntimeError("无法检测到设备")
    print("✅ ADB连接已重置")


def optimize_xiaomi_device():
    """小米设备专属优化"""
    print("⚙️ 应用小米设备优化...")
    run_adb_command(["shell", "settings", "put", "global", "hwui.disable_vsync", "1"])
    run_adb_command(["shell", "settings", "put", "global", "window_animation_scale", "0"])
    print("✅ 优化已应用")


def get_device_resolution() -> Tuple[int, int]:
    """获取设备实际分辨率"""
    try:
        output = subprocess.check_output([ADB_PATH, "shell", "wm", "size"], timeout=5)
        resolution_str = output.decode().strip().split()[-1]
        width, height = map(int, resolution_str.split('x'))
        print(f"设备实际分辨率: {width}x{height}")
        return (width, height)
    except Exception as e:
        print(f"⚠️ 无法获取设备分辨率，使用默认值: {str(e)}")
        return DEFAULT_RESOLUTION


# ==================== 视频管道类 ====================
class VideoPipeline:
    def __init__(self, resolution: Tuple[int, int]):
        self.scrcpy_proc = None
        self.ffmpeg_proc = None
        self.frame_queue = queue.Queue(maxsize=BUFFERED_FRAMES)
        self.running = False
        self.resolution = resolution
        self.frame_size = resolution[0] * resolution[1] * 3  # BGR24格式
        self.last_frame_time = 0

    def start_scrcpy(self):
        """启动scrcpy进程"""
        scrcpy_cmd = [
            SCRCPY_PATH,
            "--no-control",
            "--no-audio",
            "--max-size=1080",
            "--max-fps=30",
            "--record=-",  # 输出到stdout
            "--push-target=/sdcard/",
            "--no-cleanup",
            "--render-driver=opengl",
            "--video-codec=h264",
            "--video-bit-rate=8M",
            "--record-format=mkv"
        ]
        print("启动scrcpy命令:", " ".join(scrcpy_cmd))

        try:
            self.scrcpy_proc = subprocess.Popen(
                scrcpy_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=False
            )
            threading.Thread(target=self.monitor_scrcpy_errors, daemon=True).start()
        except Exception as e:
            print(f"❌ scrcpy启动失败: {str(e)}")
            raise RuntimeError("scrcpy启动失败")

    def monitor_scrcpy_errors(self):
        """监控scrcpy错误输出"""
        while self.running and self.scrcpy_proc:
            line = self.scrcpy_proc.stderr.readline()
            if line:
                print("scrcpy:", line.decode(errors='ignore').strip())

    def start_ffmpeg(self):
        """启动FFmpeg进程"""
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-loglevel", "debug",
            "-f", "matroska",
            "-i", "-",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "drop",
            "-fflags", "nobuffer+discardcorrupt",
            "-avioflags", "direct",
            "-max_delay", "500000",
            "-"
        ]
        print("启动FFmpeg命令:", " ".join(ffmpeg_cmd))

        try:
            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self.scrcpy_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=False
            )
            threading.Thread(target=self.monitor_ffmpeg_errors, daemon=True).start()
        except Exception as e:
            print(f"❌ FFmpeg启动失败: {str(e)}")
            raise RuntimeError("FFmpeg启动失败")

    def monitor_ffmpeg_errors(self):
        """监控FFmpeg错误输出"""
        while self.running and self.ffmpeg_proc:
            line = self.ffmpeg_proc.stderr.readline()
            if line:
                print("ffmpeg:", line.decode(errors='ignore').strip())

    def frame_reader(self):
        """帧读取线程"""
        print(f"帧读取线程启动，期望每帧大小: {self.frame_size} 字节")

        while self.running:
            try:
                # 读取完整帧数据
                raw_frame = bytearray()
                while len(raw_frame) < self.frame_size and self.running:
                    remaining = self.frame_size - len(raw_frame)
                    chunk = self.ffmpeg_proc.stdout.read(remaining)
                    if not chunk:
                        break
                    raw_frame.extend(chunk)

                if not raw_frame:
                    print("⚠️ 收到空帧，可能流已结束")
                    self.restart_pipeline()
                    continue

                if len(raw_frame) != self.frame_size:
                    print(f"⚠️ 收到不完整帧: {len(raw_frame)}/{self.frame_size} 字节")
                    self.restart_pipeline()
                    continue

                # 放入帧队列
                self.frame_queue.put(bytes(raw_frame))
                self.last_frame_time = time.time()

            except Exception as e:
                print(f"帧读取错误: {str(e)}")
                self.restart_pipeline()

    def get_frame(self, timeout=1.0) -> Optional[np.ndarray]:
        """获取一帧图像"""
        try:
            # 如果长时间没有新帧，尝试重启管道
            if time.time() - self.last_frame_time > 3.0:
                print("⚠️ 长时间无新帧，尝试重启管道...")
                self.restart_pipeline()

            raw_frame = self.frame_queue.get(timeout=timeout)

            # 将字节数据转换为numpy数组
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self.resolution[1], self.resolution[0], 3))
            return frame

        except queue.Empty:
            print("⚠️ 获取帧超时")
            return None
        except Exception as e:
            print(f"获取帧错误: {str(e)}")
            return None

    def restart_pipeline(self):
        """重启整个管道"""
        print("🔄 重启视频管道...")
        self.cleanup()
        time.sleep(1)
        self.start_scrcpy()
        self.start_ffmpeg()

    def start(self, max_retries=MAX_RETRIES):
        """启动管道"""
        self.running = True
        retries = 0

        while retries < max_retries:
            try:
                print(f"尝试启动管道 ({retries + 1}/{max_retries})...")
                self.start_scrcpy()
                time.sleep(1)  # 等待scrcpy启动
                self.start_ffmpeg()
                threading.Thread(target=self.frame_reader, daemon=True).start()
                print("✅ 视频管道启动成功")
                return
            except Exception as e:
                retries += 1
                print(f"启动失败 ({retries}/{max_retries}): {str(e)}")
                self.cleanup()
                time.sleep(2)

        raise RuntimeError(f"无法启动管道，已达到最大重试次数 {max_retries}")

    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        for proc in [self.scrcpy_proc, self.ffmpeg_proc]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.frame_queue = queue.Queue(maxsize=BUFFERED_FRAMES)  # 清空队列


# ==================== 主程序 ====================
def main():
    # 初始化
    reset_device_connection()
    optimize_xiaomi_device()

    # 获取实际分辨率
    resolution = get_device_resolution()

    # 创建视频管道
    pipeline = VideoPipeline(resolution)
    pipeline.start()

    try:
        # 创建显示窗口
        window_name = "XIAOMI 13 Ultra 投屏"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resolution[0] // 2, resolution[1] // 2)

        print("🎥 投屏已启动 (按ESC退出)...")
        frame_count = 0
        last_fps_time = time.time()

        while True:
            # 获取帧
            frame = pipeline.get_frame()

            if frame is not None:
                frame_count += 1

                # 计算FPS
                current_time = time.time()
                fps = 1 / (current_time - last_fps_time)
                last_fps_time = current_time

                # 显示信息
                cv2.putText(
                    frame, f"FPS: {int(fps)} | 帧: {frame_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                # 显示帧
                cv2.imshow(window_name, frame)
            else:
                print("⏳ 等待帧...")

            # ESC键退出
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("⏹️ 用户中断")
    finally:
        pipeline.running = False
        pipeline.cleanup()
        cv2.destroyAllWindows()
        print("🛑 投屏已停止")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 致命错误: {str(e)}")
        input("按回车键退出...")
    sys.exit()