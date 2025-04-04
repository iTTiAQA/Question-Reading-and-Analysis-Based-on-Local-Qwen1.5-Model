import subprocess
import dxcam
import cv2
import pygetwindow as gw
import time
import pyautogui
import keyboard
import OCR_identify
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import *

model_id = get_device()
print(f"设备型号: {model_id}")  # 输出: 2304FPN6DC

print("\n 正在加载模型......")
device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "./Qwen1.5-1.8B-Chat",
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained("./Qwen1.5-1.8B-Chat")
print("模型加载完成！")


# 启动投屏
process = subprocess.Popen(["scrcpy", "-m", "1024", "--max-fps", "45", "--no-audio", "--no-control"])
time.sleep(1)

# 创建摄像头对象
camera = dxcam.create()

# 获取用户输入的目标窗口标题
window_name = None
print("\n======获取窗口标题======")
if model_id:
    print(f"请确认{model_id}是否是新窗口名(是请输入yes)")
    if str(input("请输入：")).strip().lower() == 'yes':
        window_name = model_id

for _ in range(3):
    print("========请保持截屏窗口前台运行！！！=========")

if window_name is None:
    window_name = input("请输入目标窗口标题(支持部分匹配): ").strip()

# 初始获取窗口区域
region = get_window_rect_by_title(window_name)
if region is None:
    print(f"找不到标题包含 '{window_name}' 的窗口或窗口不在屏幕内, 自动结束程序")
    exit(0)


def get_response(input_prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    model_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return model_response


def press_i(img):
    global description
    choice = str(input("是否将此文件设置为题干(是请输入yes)："))
    if choice.strip().lower() == "yes":
        print("请选题目区域：")
        selector = ROISelector(img)
        img = selector.select_roi()
        desc = OCR_identify.ocr_identify(img, d_conf=30)
        print("以下是否是题干：\n")
        for paragraph in desc:
            print(paragraph)
        choice = str(input("是否将该文段设置为题干(是请输入yes)："))
        if choice.strip().lower() == "yes":
            description = "\n".join(desc)
            return

    print("请输入题干（输入<exit>结束）：")

    # 从用户输入中读取多行文章
    lines = []
    while True:
        line = str(input("请输入"))        # 读取一行输入
        if line.strip() == "<exit>":      # 结束输入
            break
        lines.append(line)

    # 将输入内容拼接成完整的文章
    description = "\n".join(lines)


def press_o(img):
    global description
    print("请选题目区域：")
    selector = ROISelector(img)
    img = selector.select_roi()
    img = Image.fromarray(img)
    print("提示词：")
    print(description)
    print("\n正在分析问题，请稍等。。。。。。")
    question = OCR_identify.ocr_identify(img)
    content = " ".join(question)
    print("获取到问题:")
    print(content)
    if description.strip().lower() != "none":
        content = "请帮我解决以下问题：\n " + description + content
    else:
        content = "请帮我解决以下问题：\n " + content
    response = get_response(content)

    print("\n=========大模型答案=========")
    print(response)


# 开始捕获
camera.start(region=region, target_fps=60)
print(f"正在捕获窗口: {window_name} (初始区域: {region})")
print("按 'q' 键退出...")

# 标志位
o_flag = True
i_flag = True
drawing = False  # 是否正在绘制矩形
ix, iy = -1, -1  # 矩形起始坐标
fx, fy = -1, -1  # 矩形结束坐标

# 记录上次区域和检查时间
last_region = region
last_check_time = time.time()
window_missing_count = 0
description = 'None'

time.sleep(0.1)
print("==========初始化结束==========")
print("按下'o'读入题目， 按下'i'录入题干\n")
if __name__ == "__main__":
    try:
        while True:
            # 每隔0.3秒检查一次窗口位置是否变化
            if time.time() - last_check_time > 0.3:
                new_region = get_window_rect_by_title(window_name)

                if new_region is None:
                    window_missing_count += 1
                    if window_missing_count > 3:  # 连续3次检测不到窗口
                        print("窗口已移出屏幕或关闭，等待窗口返回...")
                        camera.stop()
                        # 等待窗口重新出现
                        while new_region is None:
                            time.sleep(0.5)
                            new_region = get_window_rect_by_title(window_name)
                        print("窗口已恢复，重新开始捕获")
                        camera.start(region=new_region)
                        window_missing_count = 0
                elif new_region != last_region:
                    print(f"检测到窗口移动，更新捕获区域: {new_region}")
                    camera.stop()
                    camera.start(region=new_region)
                    last_region = new_region
                    window_missing_count = 0

                last_check_time = time.time()

            # 获取最新帧
            frame = camera.get_latest_frame()

            if frame is not None:
                # 变换图像格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 显示帧
                cv2.imshow("Captured Window", frame)

            # 检查是否按下了 'i' 键
            if keyboard.is_pressed('i') and i_flag:
                i_flag = False
                print("\n检测到 'i' 键按下，请输入新的指令:")
                frame = Image.fromarray(frame)
                press_i(frame)
                print("==========题干录入完毕==========")
                print("按下'o'读入题目， 按下'i'录入题干\n")
                i_flag = True

            # 检查是否按下了 'o' 键
            if keyboard.is_pressed('o') and o_flag:
                o_flag = False
                print("\n检测到'o'按下，将自动做题")
                press_o(frame)
                print("==========题目分析完毕==========")
                print("按下'o'读入题目， 按下'i'录入题干\n")
                o_flag = True

            # 检测按键
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保资源被释放
        camera.stop()
        cv2.destroyAllWindows()
        if process:
            clean_proc(process)
        print("捕获已停止")
