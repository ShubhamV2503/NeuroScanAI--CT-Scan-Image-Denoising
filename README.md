---
title: NeuroScanAI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---
# 🧠 CT Scan Image Denoising & Brain Tumor Analysis  

## 📖 Introduction  
This project focuses on **enhancing brain CT scans** by reducing acquisition noise using a **CNN-based autoencoder**, followed by **tumor detection** on the refined images.  

The workflow ensures that critical anatomical details are preserved while improving diagnostic accuracy.  
A lightweight **Flask web app** allows clinicians or researchers to upload CT scans (`.dcm` format) and instantly visualize denoised results along with tumor classification.  

Deployed seamlessly on **AWS EC2** (remote access via PuTTY).  

---

## 📂 Project Layout  

```plaintext
CT-Image-Denoising/
├── models/             # Saved deep learning models (autoencoder + classifier)
├── static/             # Assets for the Flask frontend (CSS, images, JS)
├── templates/          # HTML templates for UI (upload page, result page)
├── app.py              # Core pipeline: preprocessing, inference, evaluation
├── requirements.txt    # List of Python dependencies
└── README.md           # Documentation
```

👉 **`app.py`** acts as the entry point — handling:  
- DICOM loading & preprocessing  
- Autoencoder inference (denoising)  
- Tumor classification on denoised scans  
- Evaluation metrics (SNR, classification report)  
- Flask-based web serving  

---

## 🚀 Core Features  

- 🧠 **Noise Reduction**: Autoencoder removes CT noise while retaining diagnostic details.  
- 🩺 **Tumor Prediction**: Classifier identifies tumor presence on enhanced images.  
- 📊 **Metrics**: Includes SNR improvement tracking & classification reports.  
- 🌍 **Cloud Deployment**: Flask app hosted on **AWS EC2** for remote & Easy usage.  

---

## 📊 Performance Snapshot  

✅ **Classification Accuracy**:  
- Before denoising → **0.37**  
- After denoising → **0.84**  

✅ **Signal-to-Noise Ratio (SNR):**  

| Condition        | SNR (dB) |
|------------------|----------|
| Raw CT (noisy)   | 2.94     |
| After Denoising  | 15.58    |

---

## 🖼️ Visual Results  

### 🔹 CT Denoising Example  
_Noise vs. Enhanced Image_  
![Denoising Example](static/Picture1.png)  

---

## ⚡ Getting Started  

### 🔧 Local Setup  

1. Clone the repo & install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
2. Add trained weights (`.h5` / `.pt`) into the `models/` folder.  
3. Run the app:  
   ```bash
   python app.py
   ```  
4. Open **http://127.0.0.1:5000/** in your browser → Upload a `.dcm` scan → View results.  

---

## 👨‍💻 Author  & Co-Author
📌 **Developed by:** *Shubham Vishwakarma* / *Mayank Kathane* / *Jeet Nakrani*
📌 **Publication:** It has been Published in IEEE journal
💬 Feel free to reach out for collaboration or research discussions.  

---

✨ **In short:** This system transforms noisy CT scans into clinically useful images, leading to **better tumor detection and higher diagnostic confidence**.  
