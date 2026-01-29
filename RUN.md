# 🚀 Quick Run Guide - NeuroScan AI

## ⚡ Run in 2 Minutes

### Step 1: Start Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
✅ Server running at http://localhost:5000

### Step 2: Start Frontend (New Terminal)
```bash
cd frontend
npm install
npm start
```
✅ App running at http://localhost:3000

### Step 3: Open Browser
Go to: **http://localhost:3000**

---

## 📁 Project Structure

```
NeuroScan-AI/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_fixed.py      # Training script (MobileNetV2)
│   ├── config.yaml         # Configuration
│   ├── requirements.txt    # Python packages
│   └── models/trained/
│       └── final_model.keras   # Trained model (96.95%)
│
├── frontend/               # React web app
├── data/raw/               # MRI dataset
├── INSTRUCTIONS.md         # Detailed guide
└── RUN.md                  # This file
```

---

## 🔄 Retrain Model (Optional)

```bash
cd backend
python train_fixed.py
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Port in use | `Ctrl+C` and restart |
| Module not found | `pip install -r requirements.txt` |
| Model not loaded | Check `models/trained/final_model.keras` exists |

---

## 📊 Model Info

- **Architecture:** MobileNetV2
- **Accuracy:** 96.95%
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor

---

**That's it! Enjoy using NeuroScan AI 🧠**
