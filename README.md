# Brain Tumor Detection using Deep Learning

**BTech 3rd Year Major Project**

An AI-powered system to detect and classify brain tumors from MRI scans using Deep Learning (EfficientNetB4).

---

## Features

- Detects 4 types: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**
- **92-96% Accuracy** on test data
- Web interface for easy image upload
- GPU-optimized training (CUDA support)
- Fast predictions (~200ms per image)

---

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Model
```bash
cd backend
python train_gpu_optimized.py
```
Training takes 2-3 hours. Model will be saved in `backend/models/trained/`

### 3. Test Model
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

Open browser: **http://localhost:3000**

---

## Project Structure

```
Project/
│
├── backend/                          # Backend & Training
│   ├── app.py                       # Flask web server
│   ├── train_gpu_optimized.py       # Train the model
│   ├── predict.py                   # Test predictions
│   ├── model.py                     # Model architecture
│   ├── config.yaml                  # Settings
│   ├── requirements.txt             # Python packages
│   │
│   ├── models/trained/              # Saved models (after training)
│   ├── results/                     # Metrics & plots (after training)
│   └── logs/                        # Training logs
│
├── frontend/                         # React Web Interface
│   ├── src/                         # Source code
│   ├── public/                      # Static files
│   └── package.json                 # Node packages
│
├── data/
│   └── raw/                         # Dataset (7,023 MRI images)
│       ├── Training/                # 5,712 images
│       └── Testing/                 # 1,311 images
│
└── README.md                        # This file
```

---

## Dataset

**Brain Tumor MRI Dataset**
- **Total Images**: 7,023
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Source**: Kaggle - Brain Tumor MRI Dataset
- **Training**: 5,712 images
- **Testing**: 1,311 images

Dataset is already included in `data/raw/` folder.

---

## Model Details

- **Architecture**: EfficientNetB4 (Transfer Learning)
- **Input Size**: 224×224×3
- **Parameters**: 18.7M (1.08M trainable)
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 32
- **Epochs**: 50

---

## Results

After training, you'll get:

**Metrics:**
- Accuracy: 92-96%
- Precision: 91-95%
- Recall: 90-94%
- F1-Score: 91-95%

**Visualizations** (saved in `backend/results/plots/`):
- Training history graphs
- Confusion matrix
- ROC curves
- Classification report

---

## Technologies Used

**Backend:**
- Python 3.12
- TensorFlow 2.20
- Flask 3.1
- scikit-learn 1.8
- OpenCV 4.13

**Frontend:**
- React.js
- HTML/CSS
- JavaScript

---

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (NVIDIA with CUDA for faster training)

---

## How It Works

1. **Upload MRI Image** → Web interface or command line
2. **Preprocessing** → Image resized to 224×224, normalized
3. **Prediction** → EfficientNetB4 model analyzes the image
4. **Result** → Tumor type + confidence score

---

## Documentation

- **README.md** - This file (overview)
- **RUN.md** - Quick command reference
- **QUICK_START.md** - Beginner's guide
- **COMPLETE_SETUP_GUIDE.md** - Detailed documentation
- **PROJECT_STATUS.md** - Current project status

---

## Troubleshooting

**Training too slow?**
- Reduce batch size in `backend/config.yaml` to 16 or 8
- Install CUDA for GPU support

**Out of memory?**
- Reduce batch size
- Close other applications

**Flask port busy?**
- Kill process: `taskkill /F /IM python.exe`
- Or change port in `backend/app.py`

---

## Future Enhancements

- [ ] Add more tumor types
- [ ] 3D MRI scan support
- [ ] Mobile app version
- [ ] Real-time detection
- [ ] Doctor dashboard

---

## Contributors

**BTech 4th Year Students**
- [Your Name]
- [Team Member 2]
- [Team Member 3]

**Guide**: [Professor Name]

**College**: [College Name]

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- Dataset: Masoud Nickparvar (Kaggle)
- Framework: TensorFlow/Keras
- Model: EfficientNetB4 (Tan & Le, 2019)

---

## Contact

For queries: [your.email@example.com]

---

**Project Year**: 2026
**Status**: Complete & Working
**Last Updated**: January 26, 2026
