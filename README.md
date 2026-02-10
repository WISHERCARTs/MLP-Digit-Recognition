# 🔢 MLP Digit Recognition

ระบบจดจำตัวเลขด้วย **Multi-Layer Perceptron (Neural Network)** พัฒนาด้วย Python และ Streamlit

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)

🔗 **Live Demo:** [https://mlp-wishercarts.streamlit.app/](https://mlp-wishercarts.streamlit.app/)

## Features

- **วาดตัวเลขด้วยมือ** - Interactive canvas สำหรับวาดตัวเลข
- **อัปโหลดรูปภาพ** - รองรับไฟล์ PNG, JPG
- **Neural Network** - ใช้ MLP กับ 2 hidden layers (256, 128 neurons)
- **แสดง Confidence** - แสดงความมั่นใจในการทำนาย

## Installation

```bash
# Clone หรือ download โปรเจกต์
cd MLP

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน application
streamlit run app.py
```

## Project Structure

```
MLP/
├── app.py              # Streamlit web application
├── MLP_nural.py        # Original training script
├── requirements.txt    # Python dependencies
├── mnist-original.mat  # MNIST dataset
├── mlp_model.pkl       # Trained model (auto-generated)
└── README.md           # Documentation
```

## How MLP Works

```
Input Layer (784 neurons)
        ↓
Hidden Layer 1 (256 neurons) + ReLU
        ↓
Hidden Layer 2 (128 neurons) + ReLU
        ↓
Output Layer (10 neurons) + Softmax
        ↓
Prediction (0-9)
```

### Dataset

- **MNIST** - 70,000 handwritten digit images (28×28 pixels)
- Training: 60,000 images
- Testing: 10,000 images

## Model Performance

| Metric        | Value        |
| ------------- | ------------ |
| Accuracy      | ~97%+        |
| Hidden Layers | 2            |
| Neurons       | 256, 128     |
| Training Time | ~2-3 minutes |

## Screenshots

เมื่อรัน app จะได้หน้าเว็บสำหรับ:

1. วาดตัวเลขด้วย canvas
2. อัปโหลดรูปตัวเลข
3. ดูผลการทำนายพร้อม confidence

## 📝 License

MIT License
