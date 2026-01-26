# How to Run the Brain Tumor Detection Project

## Quick Commands

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd backend
python train_gpu_optimized.py
```
Wait 2-3 hours for training to complete.

### 3. Test the Model
```bash
cd backend
python predict.py --image ../data/raw/Testing/glioma/Te-gl_0015.jpg
```

### 4. Run Web Application

**Terminal 1 (Backend):**
```bash
cd backend
python app.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm start
```

Open: http://localhost:3000

---

## Project Structure

```
Project/
├── backend/
│   ├── app.py                     # Web server
│   ├── train_gpu_optimized.py     # Train model
│   ├── predict.py                 # Test model
│   ├── model.py                   # Model architecture
│   ├── config.yaml                # Configuration
│   └── requirements.txt           # Dependencies
│
├── frontend/                      # React web app
├── data/raw/                      # Dataset (7,023 images)
│
├── README.md                      # Overview
├── COMPLETE_SETUP_GUIDE.md        # Detailed guide
├── QUICK_START.md                 # Quick start
└── RUN.md                         # This file
```

---

## That's it! Simple and clean.
