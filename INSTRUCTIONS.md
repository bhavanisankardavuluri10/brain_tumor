# 🧠 NeuroScan AI - Brain Tumor Detection

## Complete Setup & Usage Guide

**For Students & Developers - Step-by-Step Instructions**

This guide will help you set up and run the Brain Tumor Detection project on your computer.

---

## 📋 Prerequisites

### Required Software

| Software | Version | Download Link |
|----------|---------|---------------|
| Python | 3.10+ | https://www.python.org/downloads/ |
| Node.js | 18+ LTS | https://nodejs.org/ |
| VS Code | Latest | https://code.visualstudio.com/ |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start Backend (AI Server)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

✅ You should see: `[OK] Model loaded successfully - Running on http://127.0.0.1:5000`

### Step 2: Start Frontend (Web App)

Open a **new terminal** and run:

```bash
cd frontend
npm install
npm start
```

✅ You should see: `Compiled successfully! Local: http://localhost:3000`

### Step 3: Open in Browser

Navigate to: **http://localhost:3000**

---

## 📖 Detailed Installation Guide

### 1. Install Python

1. Download Python from https://www.python.org/downloads/
2. **IMPORTANT:** During installation, check ✅ **"Add Python to PATH"**
3. Click "Install Now"

**Verify installation:**
```bash
python --version
# Should show: Python 3.10.x or higher
```

### 2. Install Node.js

1. Download Node.js LTS from https://nodejs.org/
2. Run the installer with default settings
3. Restart your terminal after installation

**Verify installation:**
```bash
node --version
# Should show: v18.x.x or higher
```

### 3. Install VS Code (Recommended)

1. Download from https://code.visualstudio.com/
2. Install with default settings
3. Open the project folder in VS Code

---

## 🏗️ Project Architecture

```
NeuroScan-AI/
├── backend/                    # Python Flask API
│   ├── app.py                  # Main server application
│   ├── model.py                # Model loading utilities
│   ├── predict.py              # Prediction logic
│   ├── train_fixed.py          # Training script (MobileNetV2)
│   ├── config.yaml             # Configuration file
│   ├── requirements.txt        # Python dependencies
│   └── models/
│       └── trained/
│           ├── final_model.keras    # Trained model (96.95% accuracy)
│           └── class_info.json      # Class labels
│
├── frontend/                   # React Web Application
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # Styles (Green/White theme)
│   │   └── index.js            # Entry point
│   └── package.json            # Node.js dependencies
│
├── data/                       # Dataset
│   └── raw/
│       ├── Training/           # Training images
│       └── Testing/            # Test images
│
├── INSTRUCTIONS.md             # This file
├── README.md                   # Project overview
└── RUN.md                      # Quick run guide
```

---

## 🤖 Model Information

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Input Size | 224 × 224 pixels |
| Training Accuracy | 96.95% |
| Classes | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| Training Strategy | 3-Phase Gradual Unfreezing |

### Detection Classes

| Class | Description | Risk Level |
|-------|-------------|------------|
| **Glioma** | Tumors from glial cells | High |
| **Meningioma** | Tumors on brain membranes | Moderate |
| **Pituitary** | Pituitary gland tumors | Moderate |
| **No Tumor** | No abnormality detected | None |

---

## 🖥️ Using the Application

### Upload & Analyze

1. **Upload MRI Scan**
   - Drag & drop an MRI image, OR
   - Click to browse files
   - Supported formats: JPG, JPEG, PNG

2. **Click "Analyze Scan"**
   - Wait for AI processing (1-3 seconds)

3. **View Results**
   - Diagnosis with confidence level
   - Probability distribution charts
   - Clinical recommendations

### Features

- ✅ Real-time AI analysis
- ✅ Interactive probability charts
- ✅ Confidence trend tracking
- ✅ Scan history
- ✅ Professional medical UI

---

## ⚠️ Troubleshooting

### Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| `'python' is not recognized` | Reinstall Python with "Add to PATH" checked |
| `Port 5000 is already in use` | Close other terminals or run `taskkill /F /IM python.exe` |
| `Port 3000 is already in use` | Close other terminals or run `taskkill /F /IM node.exe` |
| `Model not loaded` | Ensure `models/trained/final_model.keras` exists |
| `CORS error` | Make sure backend is running on port 5000 |
| `npm install fails` | Delete `node_modules` folder and try again |

### Restart Commands

**Stop all processes:**
```bash
# In each terminal, press Ctrl+C
```

**Restart backend:**
```bash
cd backend
python app.py
```

**Restart frontend:**
```bash
cd frontend
npm start
```

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status |
| `/api/predict` | POST | Upload and analyze MRI |

### Example API Call

```bash
curl -X POST -F "file=@brain_mri.jpg" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "glioma",
  "confidence": 94.5,
  "all_probabilities": {
    "glioma": 94.5,
    "meningioma": 3.2,
    "pituitary": 1.8,
    "notumor": 0.5
  }
}
```

---

## 🔧 Training the Model (Optional)

If you need to retrain the model:

```bash
cd backend
python train_fixed.py
```

**Training Features:**
- 3-phase gradual unfreezing
- Class weight balancing
- Label smoothing (0.1)
- L2 regularization + Dropout
- Data augmentation

---

## 📱 Technology Stack

### Backend
- Python 3.10+
- TensorFlow 2.16+
- Flask (Web Framework)
- MobileNetV2 (Pre-trained model)

### Frontend
- React 18
- Chart.js (Visualizations)
- Axios (HTTP Client)
- React Dropzone (File Upload)

---

## 🎓 For Presentations

### Demo Steps

1. **Before presentation:**
   ```bash
   # Terminal 1
   cd backend && python app.py
   
   # Terminal 2
   cd frontend && npm start
   ```

2. **During presentation:**
   - Show the landing page and stats
   - Upload test images from `data/raw/Testing/`
   - Explain the charts and confidence scores
   - Show history feature

3. **Key points to mention:**
   - 96.95% accuracy
   - 4 tumor types detected
   - MobileNetV2 architecture
   - Real-time analysis

---

## 📞 Support

If you encounter issues:

1. Read the error message carefully
2. Check the troubleshooting section
3. Restart both servers
4. Ensure all software is installed correctly

---

## ⚖️ Disclaimer

> **This application is for educational purposes only.**
> 
> AI predictions should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

**Last Updated:** January 2026  
**Model Accuracy:** 96.95%  
**Status:** ✅ Production Ready
