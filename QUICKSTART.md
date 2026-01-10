# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment

Run the automated setup script:

```bash
setup.bat
```

This will:

- Create a Python virtual environment
- Install all Python dependencies
- Install frontend dependencies

### Step 2: Prepare Dataset

1. Create dataset structure:

```bash
cd backend
python utils.py --create-dirs
cd ..
```

2. Add your MRI images to these folders:

   - `data/raw/glioma/` - Glioma tumor images
   - `data/raw/meningioma/` - Meningioma tumor images
   - `data/raw/pituitary/` - Pituitary tumor images
   - `data/raw/no_tumor/` - Healthy brain images

3. Check your dataset:

```bash
cd backend
python utils.py --check-dataset
cd ..
```

4. Preprocess the data:

```bash
cd backend
python data_preprocessing.py
cd ..
```

### Step 3: Train the Model

```bash
cd backend
python train.py
cd ..
```

**Note**: Training can take 20-40 minutes with GPU, or 2-4 hours with CPU.

### Step 4: Run the Application

Use the automated start script:

```bash
start.bat
```

Or manually:

**Terminal 1 - Backend:**

```bash
venv\Scripts\activate
cd backend
python app.py
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm start
```

### Step 5: Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 📝 Usage

1. Open the web interface at http://localhost:3000
2. Drag and drop or click to upload an MRI image
3. Click "Analyze Image" button
4. View the prediction results with confidence scores

## 🔧 Command-Line Tools

### Make Predictions

```bash
# Single image
cd backend
python predict.py --image path/to/image.jpg --visualize

# Batch prediction
python predict.py --directory path/to/images/
cd ..
```

### Utilities

```bash
cd backend

# Test inference speed
python utils.py --test-speed

# Generate model report
python utils.py --generate-report

# Compare preprocessing methods
python utils.py --compare-preprocessing path/to/image.jpg

cd ..
```

## 📊 Expected Results

After training, you should see:

- Training accuracy: 95%+
- Validation accuracy: 93%+
- Test accuracy: 95%+

Results are saved in `results/` directory:

- `training_history.png` - Training curves
- `confusion_matrix.png` - Classification matrix
- `roc_curves.png` - ROC curves for each class
- `evaluation_results.json` - Detailed me:

````bash
cd backend
python train.py
``

## 🐛 Common Issues

### Issue: "Model not found"

**Solution**: Train the model first using `python train.py`

### Issue: "No module named 'tensorflow'"

**Solution**: Activate virtual environment:

```bash
venv\Scripts\activate
pip install -r requirements.txt
````

### Issue: Frontend won't start

**Solution**: Install dependencies:

```bash
cd frontend
npm install
```

### Issue: Port already in use

**Solution**: Change ports in:

- Backend: `backend/config.yaml` → `api.port`
- Frontend: `frontend/package.json` → add `"start": "PORT=3001 react-scripts start"`

## 📚 Next Steps

1. **Improve Model**:

   - Add more training data
   - Adjust hyperparameters in `backend/config.yaml`
   - Try different architectures

2. **Customize UI**:

   - Edit `frontend/src/App.js` and `App.css`
   - Add new features or visualizations

3. **Deploy**:
   - See deployment guide in README.md
   - Set up production environment

## 💡 Tips

- Use GPU for faster training (20-40 min vs 2-4 hours)
- Balance your dataset for better results
- Start with a small dataset to test the pipeline
- Monitor training in TensorBoard: `tensorboard --logdir logs/`

## 🆘 Need Help?

Check the full documentation in README.md or open an issue.
