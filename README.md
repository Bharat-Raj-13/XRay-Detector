# 🏥 XRay AI — Chest X-Ray Pneumonia Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based desktop application that detects **Pneumonia** from chest X-ray images using **VGG16** transfer learning, with **Grad-CAM** heatmap visualization showing exactly which region of the lung the AI focused on.

---

## 🎯 Features

- **Real-time Pneumonia Detection** — Upload any chest X-ray and get instant results
- **Grad-CAM Heatmap** — Visual explanation showing affected lung regions in red
- **95% Accuracy** — Trained on 5,216 chest X-ray images
- **PDF Report Generation** — Save full medical report with patient details and heatmap
- **Patient History Log** — Track all previous analyses in one place
- **14-Disease Model** — Secondary DenseNet121 model detecting 14 chest diseases
- **Completely Offline** — No internet required, patient data never leaves the device
- **Professional UI** — Dark themed desktop app built with CustomTkinter

---

## 📸 Demo

| Original X-Ray | Grad-CAM Heatmap | Result |
|---|---|---|
| Chest X-Ray image | Red = affected region | PNEUMONIA — 99.8% |

---

## 🧠 Models

### Primary Model — VGG16 (95% Accuracy)
| Detail | Value |
|---|---|
| Architecture | VGG16 (pretrained on ImageNet) |
| Dataset | Chest X-Ray Images (Pneumonia) — Kaggle |
| Training images | 5,216 (1,341 Normal + 3,875 Pneumonia) |
| Epochs | 10 |
| Accuracy | ~95% |
| Classes | Normal / Pneumonia |
| Explainability | Grad-CAM |

### Secondary Model — DenseNet121 (54% Accuracy)
| Detail | Value |
|---|---|
| Architecture | DenseNet121 |
| Dataset | NIH Chest X-Ray Dataset (112,000 images) |
| Diseases | 14 chest diseases + Normal |
| Type | Multi-label classification |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow 2.16.1 / Keras** — Deep learning framework
- **VGG16** — Pretrained CNN for feature extraction
- **Grad-CAM** — Explainability / heatmap generation
- **CustomTkinter** — Professional desktop UI
- **OpenCV** — Image processing
- **ReportLab** — PDF report generation
- **NumPy / Pillow** — Data processing

---

## 📁 Project Structure
XRay-Detector/
├── main_app.py          ← Main app (VGG16, 95% accuracy)
├── app_nih.py           ← 14-disease app (DenseNet121)
├── train_model.py       ← VGG16 training script
├── train_model_nih.py   ← DenseNet121 training script
├── model/
│   ├── xray_model.h5    ← Trained VGG16 model
│   └── xray_nih_model.h5← Trained DenseNet121 model
└── README.md
---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11
- Windows OS

### Step 1 — Clone the repository
```bash
git clone https://github.com/Bharat-Raj-13/XRay-Detector.git
cd XRay-Detector
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install tensorflow==2.16.1 customtkinter opencv-python numpy pillow reportlab scikit-learn matplotlib
```

### Step 4 — Download the dataset
Download the Chest X-Ray dataset from Kaggle:
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract to: `XRay-Detector/chest_xray/`

### Step 5 — Run the app
```bash
python main_app.py
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~96% |
| Validation Accuracy | ~95% |
| Model Size | 57MB |
| Inference Time | < 1 second |
| Dataset Size | 5,216 images |

---

## 🔬 How It Works
Chest X-Ray Image
↓
Preprocessing (224×224, normalize)
↓
VGG16 Feature Extraction (13 conv layers)
↓
Custom Classification Head
(GAP → Dense 256 → Dropout → Dense 1)
↓
Sigmoid Output
↓
Normal / Pneumonia + Confidence Score
↓
Grad-CAM Heatmap Generation
---

## 📄 Research Papers

1. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556
2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* arXiv:1610.02391
3. Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection.* arXiv:1711.05225
4. Huang, G., et al. (2017). *Densely Connected Convolutional Networks.* arXiv:1608.06993
5. Wang, X., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database.* arXiv:1705.02315

---

## ⚠️ Disclaimer

This application is for educational purposes only. It should not replace professional medical diagnosis. Always consult a qualified doctor for medical advice.

---

## 👥 Team

- BTech AI Students
- Project: Edge AI / Medical Imaging
- Institution: BTech Artificial Intelligence

---

## 📜 License

This project is licensed under the MIT License.