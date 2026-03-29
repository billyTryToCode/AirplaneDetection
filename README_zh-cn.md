#  Airplane Detection Project

- [English](README.md)
- [中文](README_zh-cn.md) (你在这！)

#  飞机检测项目（Faster R-CNN & YOLO）

本项目演示如何使用以下模型对图片中的飞机进行检测：
- Faster R-CNN（ResNet50 FPN V2）
- YOLO（Ultralytics，支持 YOLOv8 / YOLO11 / YOLO26 等多个版本）

---

##  环境

- GPU：RTX 4060
- CUDA：13.0
- 操作系统：Linux（Ubuntu20.04）

⚠️ 注意：
- 本项目**可以在 CPU 上运行**，但速度会较慢。
- 如果你想使用自己的 CUDA 版本，请根据官方说明自行安装对应的 PyTorch。

---

##  1. 安装 Conda（如果尚未安装）

```bash
# 下载 Miniconda（Linux 示例）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重启终端或执行
source ~/.bashrc
````

---

##  2. 创建 Python 环境（Python 3.10）

```bash
conda create -n airplane-detect python=3.10 -y
conda activate airplane-detect
```

⚠️ 请确保在继续之前已经激活该环境。

---

##  3. 安装 PyTorch

```bash
pip3 install torch torchvision
```

---

##  4. 验证 PyTorch 安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

期望输出：

* 能正确打印版本号
* 如果 GPU 配置正确，CUDA 应显示为 `True`

---

##  5. 安装其他依赖

```bash
pip install opencv-python matplotlib pillow tqdm
```

---

##  6. 安装 YOLO（Ultralytics）

```bash
pip install ultralytics
```

---

##  7. 验证 YOLO 安装

```bash
yolo help
```

如果没有报错，说明安装成功。

---

##  项目结构

```
.
├── images/                     # 输入图片（10张飞机图片）
├── outputs/                    # 输出结果（运行时会被覆盖）
├── detect_airplane.py          # Faster R-CNN 检测脚本
├── detect_airplane_YOLO.py     # YOLO 检测脚本
└── README.md
```

---

##  重要说明

* `outputs/` 文件夹中已经包含之前的运行结果
* 程序运行时会**覆盖该文件夹**

如果你想保留之前的结果：

```bash
cp -r outputs outputs_backup
```

---

##  使用的模型

### 1. Faster R-CNN

* 模型：`FasterRCNN_ResNet50_FPN_V2`
* 使用 COCO 数据集预训练
* 可在代码中调整阈值（threshold）

---

### 2. YOLO

* 基于 Ultralytics YOLO

* 支持切换不同模型：

  * `yolov8n`
  * `yolo11n`
  * `yolo26n`

* 阈值（threshold）可在代码中调整

---

##  8. 运行检测

### Faster R-CNN

```bash
python ./detect_airplane.py
```

---

### YOLO

```bash
python ./detect_airplane_YOLO.py
```

---

##  输出结果

* 所有结果保存在：

```
./outputs/
```

---

##  最佳结果

| 模型           | Threshold | 检测到的飞机数量 |
| ------------ | --------- | -------- |
| Faster R-CNN | 0.30      | 46       |
| YOLO26n      | 0.25      | 48       |

---

##  重新运行说明

* 你可以修改：

  * 检测阈值（threshold）
  * YOLO 模型版本
* 然后重新运行脚本进行对比

---

##  常见问题（Troubleshooting）

### CUDA 无法使用？

请检查：

* `torch.cuda.is_available()` 是否返回 True
* 显卡驱动与 CUDA 是否匹配

---

### YOLO 命令无法使用？

```bash
pip install ultralytics --upgrade
```

---