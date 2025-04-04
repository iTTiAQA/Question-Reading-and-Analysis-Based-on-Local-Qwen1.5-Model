import pytesseract
from PIL import Image
import cv2
import numpy as np
import time

# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(img):
    """图像预处理增强"""
    # 使用numpy
    img_arr = np.array(img)
    # cv2.imshow("Binary", img_arr)
    # cv2.waitKey(0)

    # 获取图像的宽度、高度和通道数
    height, width, channel = img_arr.shape

    # 缩放图像
    # 将图像宽度和高度都缩小为原来的一半
    new_width = int(width * 1.5)
    new_height = int(height * 1.5)
    # img_arr = cv2.resize(img_arr, (new_width, new_height), cv2.INTER_AREA)

    # 转换灰度图
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_arr = clahe.apply(img_arr)

    # 中值滤波降噪
    # img_arr = cv2.medianBlur(img_arr, 3)

    _, binary_img = cv2.threshold(img_arr, 200, 255, cv2.THRESH_BINARY)
    # 二值化处理
    # 由于背景接近白色，使用阈值分割，将背景设置为白色（255），前景设置为黑色（0）
    cv2.imshow("Binary", binary_img)
    cv2.waitKey(0)

    return Image.fromarray(binary_img)


def get_dynamic_gap(data, img_size, verbose=False):
    """
    优化版动态间距阈值计算
    :param data: Tesseract OCR输出的字典数据
    :param img_size: 图像尺寸 (width, height)
    :param verbose: 是否打印调试信息
    :return: (paragraph_threshold, line_spacing_threshold)
    """
    # 初始化统计容器
    heights = []
    line_gaps = []
    prev_bottom = None

    # 第一次遍历：收集基础数据
    for i in range(len(data['text'])):
        if not data['text'][i].strip() or int(data['conf'][i]) < 10:
            continue

        height = data['height'][i]
        top = data['top'][i]
        bottom = top + height

        heights.append(height)

        if prev_bottom is not None:
            gap = top - prev_bottom

            if 0 < gap < img_size[1] * 0.2:  # 过滤异常大间距
                line_gaps.append(gap)

        prev_bottom = bottom

    # 异常值过滤
    def filter_outliers(values, m=2):
        if not values:
            return values
        median = np.median(values)
        return [x for x in values if abs(x - median) < m * np.std(values)]

    # 计算核心统计量
    filtered_heights = filter_outliers(heights)
    filtered_gaps = filter_outliers(line_gaps)

    avg_height = np.mean(filtered_heights) if filtered_heights else img_size[1] * 0.02
    median_gap = np.median(filtered_gaps) if filtered_gaps else avg_height * 1.2

    # 动态权重计算
    density = len(data['text']) / (img_size[0] * img_size[1]) if img_size[0] * img_size[1] > 0 else 0
    density_weight = 1.5 + (0.5 / (1 + np.exp(-10 * (density - 0.001))))  # sigmoid函数调整

    # 最终阈值计算
    paragraph_threshold = max(
        median_gap * density_weight,
        avg_height * 2.3,
        img_size[1] * 0.03  # 最小保障阈值
    )

    # 行间距阈值（用于后续行合并判断）
    line_spacing_threshold = min(
        max(avg_height * 0.8, median_gap * 0.7),
        avg_height * 1.2
    )

    if verbose:
        print(f"统计报告：")
        print(f"平均字高: {avg_height:.1f}px | 中值行距: {median_gap:.1f}px")
        print(f"文本密度: {density:.6f} | 动态权重: {density_weight:.2f}")
        print(f"段落阈值: {paragraph_threshold:.1f}px | 行距阈值: {line_spacing_threshold:.1f}px")

    return paragraph_threshold, line_spacing_threshold


def reorganize_paragraph(data, lang, img_size, dconf, dynamic_gap=False):
    # 步骤2：重组段落
    paragraphs = []
    current_para = []
    current_conf = []
    prev_bottom = None

    # 获取动态阈值
    if dynamic_gap:
        para_thresh, line_thresh = get_dynamic_gap(data, img_size, verbose=False)

    for i in range(len(data['text'])):
        text = data['text'][i]
        conf = int(data['conf'][i])

        # 调整过滤阈值
        if conf < dconf or not text.strip():
            continue

        # 获取位置信息
        top, height = data['top'][i], data['height'][i]
        bottom = top + height

        # 段落判断逻辑调整(中文段落间距可能不同)
        if prev_bottom is not None:
            line_gap = top - prev_bottom

            # 判断是否使用动态行距
            if dynamic_gap:
                paragraph_threshold = 2 * line_thresh
            else:
                paragraph_threshold = height * 3
            is_new_para = line_gap > paragraph_threshold

            if is_new_para and current_para:
                # 中文置信度阈值降低到30
                if sum(current_conf) / len(current_conf) >= 30:
                    # 中文不需要按空格拼接
                    para_text = ''.join(current_para) if lang == 'chi_sim' else ' '.join(current_para)
                    paragraphs.append(para_text)
                current_para = []
                current_conf = []

        current_para.append(text.strip())
        current_conf.append(conf)
        prev_bottom = bottom

    # 处理最后一个段落
    if current_para:
        if sum(current_conf) / len(current_conf) >= 30:
            para_text = ''.join(current_para) if lang == 'chi_sim' else ' '.join(current_para)
            paragraphs.append(para_text)

    return paragraphs


def ocr_identify(img, lang='chi_sim+eng', d_conf=8):
    """
    支持中文的段落识别OCR
    :param img: 输入图像(PIL Image或路径)
    :param lang: 使用的语言包(chi_sim=简体中文, eng=英文)
    :return: 段落列表
    """
    # img = preprocess_image(img)

    # 步骤1：获取带布局信息的OCR结果
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config=f'--psm 6 -c preserve_interword_spaces=1',
        lang=lang  # 添加语言参数
    )

    image_size = tuple((img.width, img.height))
    paragraphs = reorganize_paragraph(data, lang, image_size, d_conf)

    return paragraphs


if __name__ == "__main__":
    start_time = time.time()
    # 使用示例
    image_path = "screenshot.jpg"
    image = Image.open(image_path)
    paragraphs = ocr_identify(image)

    for i, para in enumerate(paragraphs, 1):
        print(f"段落 {i}: ")
        print(para)
        print("-" * 40)

    print(f"总耗时：{time.time() - start_time}")
