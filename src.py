import subprocess
import dxcam
import cv2
import pygetwindow as gw
import time
import pyautogui  # 用于获取屏幕尺寸


def get_device():
    result = subprocess.run(
        ["adb", "devices", "-l"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.split("\n"):
        if "device" in line and "model:" in line:
            parts = line.split()
            for part in parts:
                if part.startswith("model:"):
                    return part.split(":")[1]
    return None


def get_window_rect_by_title(window_title):
    """获取窗口当前的位置和大小，并确保在屏幕范围内"""
    try:
        window = gw.getWindowsWithTitle(window_title)[0]

        # 获取屏幕尺寸
        screen_width, screen_height = pyautogui.size()

        # 确保窗口坐标在屏幕范围内
        left = max(0, window.left)
        top = max(0, window.top)
        right = min(screen_width, window.right)
        bottom = min(screen_height, window.bottom)

        # 如果窗口完全移出屏幕，返回None
        if left >= right or top >= bottom:
            return None

        return left, top, right, bottom

    except IndexError:
        return None


def clean_proc(proc):
    """清理资源"""
    print("🧹 清理资源...")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


class ROISelector:
    def __init__(self, image):
        self.image = image
        if self.image is None:
            raise ValueError("无法加载图像，请检查路径")
        self.clone = self.image.copy()
        self.roi = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1

    def select_roi(self):
        cv2.namedWindow("choose ROI")
        cv2.imshow("choose ROI", self.clone)
        cv2.setMouseCallback("choose ROI", self._mouse_callback)

        while True:
            cv2.imshow("choose ROI", self.clone)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER退出
                break
            elif key == ord('r'):  # 重置
                self.clone = self.image.copy()
                self.roi = None

        cv2.destroyWindow("choose ROI")
        return self.roi

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = max(0, x), max(0, y)  # 确保不小于0
            self.clone = self.image.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp = self.image.copy()
                x = max(0, min(x, self.image.shape[1] - 1))  # 限制在宽度范围内
                y = max(0, min(y, self.image.shape[0] - 1))  # 限制在高度范围内
                cv2.rectangle(temp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                self.clone = temp

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            fx = max(0, min(x, self.image.shape[1] - 1))  # 限制在宽度范围内
            fy = max(0, min(y, self.image.shape[0] - 1))  # 限制在高度范围内

            x1, y1 = min(self.ix, fx), min(self.iy, fy)
            x2, y2 = max(self.ix, fx), max(self.iy, fy)

            # 确保ROI有有效尺寸
            if x2 > x1 and y2 > y1:
                self.roi = self.image[y1:y2, x1:x2]
                self.coordinates = (x1, y1, x2, y2)
                cv2.rectangle(self.clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
