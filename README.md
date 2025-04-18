# 🤟 ASL Sign Language Detection using MobileNetV2

This project focuses on detecting American Sign Language (ASL) hand gestures using deep learning. It includes data collection with OpenCV and MediaPipe, training a MobileNetV2-based image classification model, and evaluating its performance on a custom dataset of ASL signs (0-9 and A-Z).

---

## 🛠️ Installation
```bash
git clone https://github.com/therealsheero/ASL-Detection.git
cd ASL-Detection
```

## 📂 Project Structure
```
.
├── data/                  # Dataset (A-Z/0-9 folders)
├── asl_mobilenetv2_best.pth  #Model
├── Collec_Data.py         # Data collection script
├── train_pth.ipynb        # Model training
├── test.py             # Real-time detection
└── requirements.txt       # Dependencies
```

## 🖐️ Data Collection
```bash
python Collect_Data.py 
```
**Controls**:
- `S` - Save frame
- `Q` - Quit
- Auto-cropping to hand region

data/
├── A/
│   ├── image1.png
│   ├── image2.png
├── B/
│   ├── image1.png
...


## 🧠 Model Training
```bash
python train_pth.ipynb 
  --data_dir data 
  --model mobilenetv2 
  --epochs 20 
  --output asl_mobilenetv2_best.pth
```

**Training Results**:
```
Epoch 20/20 | Epoch 20: Train Acc: 1.0000, Val Acc: 0.9722
Test Accuracy: 98.3%
```

## ▶️ Real-Time Detection
```bash
python test.py 
```

## 📊 Performance
| Metric       | Value |
|--------------|-------|
| Accuracy     | 98.3% |

## 🌟 Key Files
- `Collect_Data.py`: Hand tracking + data saver
- `train_pth.ipynb`: Model training pipeline
- `test.py`: Live webcam detection

## 🤝 Contributing
1. Fork the repository
2. Add more ASL samples
3. Submit a pull request

## 📧 Contact
[2004divyanshii@gmail.com](mailto:2004divyanshii@gmail.com)
```
▶️ How to Use
This model can be integrated into a real-time webcam-based ASL interpreter using OpenCV and MediaPipe or cvzone. Load the model, capture hand ROI, preprocess it, and run predictions.

🚀 Future Work
Add real-time ASL detection app
Build ASL-based games (e.g., ASL crossword)
Improve dataset diversity
Deploy on web or mobile using TensorFlow Lite or ONNX
```




