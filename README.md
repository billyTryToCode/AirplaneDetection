- [English](README.md) (You are here!)
- [中文](README_zh-cn.md)

#  Airplane Detection Project (Faster R-CNN & YOLO)

This project demonstrates how to detect airplanes in images using:
- Faster R-CNN (ResNet50 FPN V2)
- YOLO (Ultralytics, supports multiple versions like YOLOv8 / YOLO11 / YOLO26)

---

##  Environment

- GPU: RTX 4060
- CUDA: 13.0
- OS: Linux (Ubuntu20.04)

⚠️ Notes:
- The project **can run on CPU**, but will be slower.
- If you want to use your own CUDA version, please install PyTorch accordingly from official instructions.

---

##  1. Install Conda (if not installed)

```bash
# Download Miniconda (example for Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Restart shell or run
source ~/.bashrc
````

---

##  2. Create Python Environment (Python 3.10)

```bash
conda create -n airplane-detect python=3.10 -y
conda activate airplane-detect
```

⚠️ Always ensure you activated the environment before continuing.

---

##  3. Install PyTorch

```bash
pip3 install torch torchvision
```

---

##  4. Verify PyTorch Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

* Version prints correctly
* CUDA shows `True` (if GPU configured correctly)

---

##  5. Install Other Dependencies

```bash
pip install opencv-python matplotlib pillow tqdm
```

---

##  6. Install YOLO (Ultralytics)

```bash
pip install ultralytics
```

---

##  7. Verify YOLO Installation

```bash
yolo help
```

If no error appears, installation is successful.

---

##  8. Clone the repository

```bash
git clone https://github.com/billyTryToCode/AirplaneDetection.git
```

And then change the directory to AirplaneDetection.

```bash
cd AirplaneDetection
```

---

##  Project Structure

```
.
├── images/                     # Input images (10 airplane images)
├── outputs/                    # Output results (will be overwritten)
├── detect_airplane.py          # Faster R-CNN detection script
├── detect_airplane_YOLO.py     # YOLO detection script
└── README.md
```

---

##  Important Notes

* Make sure you are in the directory to run the commands!
* The `outputs/` folder already contains previous results.
* Running the program will **overwrite this folder**.

If you want to keep previous results:

```bash
cp -r outputs outputs_backup
```

---

##  Models Used

### 1. Faster R-CNN

* Model: `FasterRCNN_ResNet50_FPN_V2`
* Pretrained on COCO
* Threshold adjustable in code

---

### 2. YOLO

* Based on Ultralytics YOLO
* Supports switching models:

  * `yolov8n`
  * `yolo11n`
  * `yolo26n`

* Threshold adjustable in code

---

##  9. Run Detection

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

##  Output

* All results will be saved in:

```
./outputs/
```

---

##  Results (Best)

| Model        | Threshold | Detected Airplanes |
| ------------ | --------- | ------------------ |
| Faster R-CNN | 0.30      | 46                 |
| YOLO26n      | 0.25      | 48                 |

---

##  Notes on Re-running

* You can modify:

  * Detection threshold
  * YOLO model version
* Re-run scripts to compare results

---

##  Troubleshooting

### CUDA not working?

* Make sure:

  * `torch.cuda.is_available()` returns True
  * Driver & CUDA compatible

### YOLO command not found?

```bash
pip install ultralytics --upgrade
```

---