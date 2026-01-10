# Backend - Brain Tumor Detection System

This directory contains all the backend Python code for the Brain Tumor Detection System.

## 📁 Files

- **`model.py`** - CNN architecture (EfficientNet-B4, ResNet, Inception)
- **`data_preprocessing.py`** - Data loading and preprocessing pipeline
- **`train.py`** - Model training script with two-phase training
- **`app.py`** - Flask REST API server
- **`predict.py`** - Command-line prediction utility
- **`utils.py`** - Utility functions and tools
- **`test_system.py`** - Automated testing suite
- **`config.yaml`** - Configuration file

## 🚀 Quick Start

### 1. Preprocess Data

```bash
run_preprocessing.bat
# OR
python data_preprocessing.py
```

### 2. Train Model

```bash
run_training.bat
# OR
python train.py
```

### 3. Start API Server

```bash
run_api.bat
# OR
python app.py
```

### 4. Make Predictions

```bash
run_predict.bat --image path/to/image.jpg --visualize
# OR
python predict.py --image path/to/image.jpg
```

## 📊 API Endpoints

When the API server is running (port 5000):

- `GET /api/health` - Health check
- `POST /api/predict` - Single image prediction
- `POST /api/predict/batch` - Batch predictions
- `GET /api/classes` - Get available classes
- `GET /api/stats` - Get statistics
- `GET /api/model/info` - Model information

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Model architecture
- Training hyperparameters
- Data augmentation settings
- API configuration
- File paths

## 🧪 Testing

```bash
python test_system.py
```

## 📝 Notes

- All paths in `config.yaml` are relative to the backend directory
- Make sure to activate the virtual environment before running scripts
- Model will be saved in `../models/` directory
- Results and visualizations are saved in `../results/`
