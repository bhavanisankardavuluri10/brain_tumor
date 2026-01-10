# Project Structure

## 📁 Reorganized Directory Structure

The project is now properly organized with **frontend** and **backend** in separate folders:

```
Advanced Brain Tumor Detection in MRI Images/
│
├── 📂 backend/                          # Backend Python Application
│   ├── model.py                        # CNN model architecture
│   ├── data_preprocessing.py           # Data pipeline
│   ├── train.py                        # Training script
│   ├── app.py                          # Flask API server
│   ├── predict.py                      # Prediction utility
│   ├── utils.py                        # Utility functions
│   ├── test_system.py                  # Testing suite
│   ├── config.yaml                     # Configuration
│   ├── run_preprocessing.bat           # Run preprocessing
│   ├── run_training.bat                # Run training
│   ├── run_api.bat                     # Start API
│   ├── run_predict.bat                 # Run predictions
│   └── README.md                       # Backend docs
│
├── 📂 frontend/                         # Frontend React Application
│   ├── public/
│   │   ├── index.html                  # HTML template
│   │   └── manifest.json               # PWA manifest
│   ├── src/
│   │   ├── App.js                      # Main component
│   │   ├── App.css                     # Styling
│   │   ├── index.js                    # Entry point
│   │   └── index.css                   # Global styles
│   ├── package.json                    # Node dependencies
│   └── node_modules/                   # Node packages
│
├── 📂 data/                             # Dataset Storage
│   ├── raw/                            # Original images
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   └── processed/                      # Preprocessed data
│       └── processed_data.npz
│
├── 📂 models/                           # Saved Models
│   └── best_model.h5                   # Trained model
│
├── 📂 results/                          # Training Results
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── predictions_visualization.png
│   └── evaluation_results.json
│
├── 📂 logs/                             # Training Logs
│   └── (TensorBoard logs)
│
├── 📂 uploads/                          # Temporary Uploads
│   └── (auto-cleaned)
│
├── 📄 requirements.txt                  # Python packages
├── 📄 .gitignore                        # Git ignore rules
├── 📄 LICENSE                           # License file
├── 📄 README.md                         # Main documentation
├── 📄 QUICKSTART.md                     # Quick start guide
├── 📄 PROJECT_SUMMARY.md                # Project overview
├── 📄 API_TESTING.md                    # API testing guide
├── 📄 DEPLOYMENT.md                     # Deployment guide
├── 📄 OVERVIEW.txt                      # Quick reference
├── 🔧 setup.bat                         # Setup script
├── 🔧 start.bat                         # Start script
└── 🔧 verify.bat                        # Verification script
```

## 🔄 Changes Made

### ✅ Backend Files Moved

All Python backend files are now in the `backend/` folder:

- `model.py` → `backend/model.py`
- `data_preprocessing.py` → `backend/data_preprocessing.py`
- `train.py` → `backend/train.py`
- `app.py` → `backend/app.py`
- `predict.py` → `backend/predict.py`
- `utils.py` → `backend/utils.py`
- `test_system.py` → `backend/test_system.py`
- `config.yaml` → `backend/config.yaml`

### ✅ Configuration Updated

- `backend/config.yaml` - All paths updated to use `../` prefix
- Scripts updated to work with new structure

### ✅ New Backend Scripts

Added convenient scripts in `backend/` folder:

- `run_preprocessing.bat` - Run data preprocessing
- `run_training.bat` - Train the model
- `run_api.bat` - Start API server
- `run_predict.bat` - Make predictions
- `README.md` - Backend documentation

### ✅ Root Scripts Updated

- `setup.bat` - Updated for new structure
- `start.bat` - Updated to start from correct directories
- `verify.bat` - Updated to check new structure

### ✅ Documentation Updated

- `README.md` - Reflects new structure
- `QUICKSTART.md` - Updated commands
- Added `backend/README.md`

## 🚀 How to Use

### Running Backend Commands

**Option 1: Use the convenient batch files in backend/**

```bash
cd backend
run_preprocessing.bat    # Preprocess data
run_training.bat          # Train model
run_api.bat               # Start API server
run_predict.bat --image test.jpg --visualize
```

**Option 2: Run directly with Python**

```bash
cd backend
python data_preprocessing.py
python train.py
python app.py
python predict.py --image path/to/image.jpg
```

### Running Frontend

```bash
cd frontend
npm start
```

### Running Complete System

From the root directory:

```bash
start.bat    # Starts both backend and frontend
```

## 📝 Path Configuration

All paths in `backend/config.yaml` use relative paths from the backend directory:

```yaml
data:
  raw_data_path: "../data/raw"
  processed_data_path: "../data/processed"

paths:
  model_save_path: "../models"
  logs_path: "../logs"
  results_path: "../results"

api:
  upload_folder: "../uploads"
```

## 🎯 Benefits of New Structure

✅ **Better Organization** - Clear separation of concerns
✅ **Easier Navigation** - Related files grouped together
✅ **Professional Structure** - Industry-standard layout
✅ **Easier Deployment** - Can deploy backend and frontend separately
✅ **Better Scalability** - Easy to add more components
✅ **Cleaner Root** - Less clutter in main directory

## 🔧 Quick Reference

### Backend Operations

```bash
cd backend
python data_preprocessing.py   # Preprocess
python train.py                 # Train
python app.py                   # API server
python predict.py              # Predict
python utils.py --check-dataset # Utilities
```

### Frontend Operations

```bash
cd frontend
npm install    # Install
npm start      # Develop
npm run build  # Production build
```

### System Operations

```bash
setup.bat      # Setup everything
verify.bat     # Verify installation
start.bat      # Start all services
```

## 📚 Documentation

- **Root README.md** - Main documentation
- **backend/README.md** - Backend-specific guide
- **QUICKSTART.md** - Quick start instructions
- **API_TESTING.md** - API testing examples
- **DEPLOYMENT.md** - Deployment guide

All documentation has been updated to reflect the new structure!
