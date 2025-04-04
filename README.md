# Question-Reading-and-Analysis-Based-on-Local-Qwen1.5-Model
# 基于 Qwen1.5 的智能题目解答系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Qwen](https://img.shields.io/badge/Qwen-1.5-1.8B-green.svg)

本项目是一个集成了屏幕捕获、OCR识别和本地大语言模型的智能题目解答系统。系统通过实时捕获手机投屏窗口，结合OCR技术提取题目内容，并利用本地部署的Qwen1.5-1.8B-Chat模型进行题目分析和解答。
以下是针对您的项目优化的 `README.md` 文件，包含完整的功能说明、安装指南和使用方法：
通过屏幕捕获和OCR技术识别题目，并利用本地Qwen大模型进行智能解答的教育辅助工具。

## ✨ 功能特性

- 📱 实时手机投屏捕获（通过scrcpy）
- 🔍 交互式题目区域选择（鼠标框选）
- ✂️ 高精度OCR文字识别（支持多语言）
- 🧠 本地Qwen1.5-1.8B大模型推理
- 📚 题干记忆功能（支持多题关联）
- ⌨️ 快捷键操作（i录入题干/o解答题目）

## 🛠️ 安装指南

### 前置要求
- NVIDIA GPU + CUDA 11.7+
- Windows 10/11 或 Linux
- Android手机（需开启USB调试）

### 1. 安装依赖
```bash
git clone https://github.com/your-repo/screen-qa-system.git
cd screen-qa-system
pip install -r requirements.txt
```

### 2. 下载模型权重
SDK下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen1.5-1.8B-Chat')

Git下载
请确保 lfs 已经被正确安装
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen1.5-1.8B-Chat.git

> ⚠️ 需遵守 [Tongyi千问研究许可证](./LICENSE-TongyiQianwen.txt)

### 3. 配置ADB环境
1. 安装 [scrcpy](https://github.com/Genymobile/scrcpy)
2. 手机开启USB调试模式

## 🚀 使用说明

### 启动程序
```bash
python main.py
```

### 操作流程
1. 按提示输入/确认手机投屏窗口名
2. **录入题干**：
   - 按 `i` 键 → 选择区域或手动输入
3. **解答题目**：
   - 按 `o` 键 → 框选题目区域
4. 按 `q` 键退出

### 快捷键说明
| 按键 | 功能 |
|------|------|
| i    | 录入/更新题干 |
| o    | 解答选定题目 |
| q    | 退出程序 |

## 📂 项目结构
```
.
├── Qwen1.5-1.8B-Chat/    # 模型权重（需自行下载）
├── src.py                # 核心代码
├── OCR_identify.py       # OCR识别模块
├── main.py               # 主程序
└── requirements.txt      # 依赖列表
```

## 📜 许可证
- 代码部分：MIT License ([LICENSE](./LICENSE))
- Qwen模型：[Tongyi千问研究许可证](./LICENSE-TongyiQianwen)（**仅限非商业用途**）

## 💡 常见问题

### Q: 窗口捕获失败怎么办？
A: 请确认：
1. scrcpy正常运行 `scrcpy -m 1024`
2. 窗口标题包含设备型号（如"2304FPN6DC"）

### Q: OCR识别不准确？
A: 尝试：
1. 调整框选区域
2. 修改`OCR_identify.py`中的`d_conf`参数

### Q: 如何提高推理速度？
A: 可尝试：
```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"  # 启用FlashAttention
)
```

> 商业用途需联系阿里云授权：license@alibabacloud.com
```

---
