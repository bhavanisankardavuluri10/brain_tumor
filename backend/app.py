"""
Flask API Backend for Brain Tumor Detection with Gemini AI Report Generation
Provides RESTful endpoints for image upload, prediction, and medical report generation
"""

import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import json
import uuid

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras

# Removed dependencies - simplified for BTech project


class BrainTumorAPI:
    """
    Enhanced API class for brain tumor detection with Gemini AI reports
    """

    def __init__(self, config_path='config.yaml'):
        self.app = Flask(__name__)
        CORS(self.app)

        # Get the directory where app.py is located
        self.base_dir = Path(__file__).parent.absolute()

        # Load configuration
        config_full_path = self.base_dir / config_path
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup
        self.model = None
        self.upload_folder = self.base_dir / self.config['api']['upload_folder']
        os.makedirs(self.upload_folder, exist_ok=True)

        # Initialize class names from config (will be overridden if class_info.json exists)
        self.class_names = self.config['classes']

        # Initialize (simplified)
        self.gemini_generator = None

        # Load model
        self.load_model()

        # Register routes
        self.register_routes()

        # Statistics
        self.prediction_count = 0
        self.predictions_history = []

    def load_model(self):
        """Load the trained model (checks multiple locations)"""

        # Try multiple model paths
        model_paths = [
            self.base_dir / 'models/trained/final_model.keras',
            self.base_dir / 'models/trained/brain_tumor_model_best.keras',
            self.base_dir / f"{self.config['paths']['model_save_path']}/best_model.h5",
            self.base_dir / f"{self.config['paths']['model_save_path']}/best_model.keras"
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                try:
                    self.model = keras.models.load_model(model_path)
                    print(f"[OK] Model loaded successfully from {model_path}")

                    # Load class info if available
                    class_info_path = self.base_dir / 'models/trained/class_info.json'
                    if os.path.exists(class_info_path):
                        with open(class_info_path, 'r') as f:
                            class_info = json.load(f)
                            self.class_names = class_info['classes']
                            print(f"[OK] Class info loaded: {self.class_names}")
                    return
                except Exception as e:
                    print(f"[ERROR] Error loading model from {model_path}: {e}")
                    continue

        print("="*70)
        print("[WARNING] No trained model found!")
        print("="*70)
        print("Please train the model first by following these steps:")
        print("1. Download dataset: See DATASET_SETUP.md")
        print("2. Place data in data/raw/ folder")
        print("3. Run: python backend/train_complete.py")
        print("="*70)

    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        allowed_extensions = self.config['api']['allowed_extensions']
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img_size = tuple(self.config['model']['input_shape'][:2])
        img = cv2.resize(img, img_size)

        # Normalize
        img = img.astype('float32') / 255.0

        # Expand dimensions for batch
        img_array = np.expand_dims(img, axis=0)

        return img_array, img

    def predict_image(self, image_path):
        """Make prediction on image"""
        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded. Please train the model first."
            }

        try:
            # Preprocess
            img_array, original_img = self.preprocess_image(image_path)

            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            class_probabilities = predictions[0]

            # Get results
            predicted_class_idx = np.argmax(class_probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(class_probabilities[predicted_class_idx]) * 100

            # Get top 3 predictions
            top_k = min(3, len(self.class_names))
            top_indices = np.argsort(class_probabilities)[-top_k:][::-1]

            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(class_probabilities[idx]) * 100
                }
                for idx in top_indices
            ]

            # Determine if confident prediction
            threshold = self.config['prediction']['confidence_threshold'] * 100
            is_confident = confidence >= threshold

            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'is_confident': is_confident,
                'all_probabilities': {
                    self.class_names[i]: round(float(class_probabilities[i]) * 100, 2)
                    for i in range(len(self.class_names))
                },
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }

            # Add warning for tumor cases
            if predicted_class != 'notumor' and is_confident:
                result['warning'] = f"Tumor detected: {predicted_class}. Please consult a medical professional."

            # Generate Gemini AI Medical Report
            # Removed Gemini integration
            if False:  # Disabled
                try:
                    medical_report = None
                    result['medical_report'] = medical_report
                    result['ai_report_available'] = True
                    print("[OK] Gemini medical report generated")
                except Exception as e:
                    print(f"[WARNING] Gemini report generation failed: {e}")
                    result['ai_report_available'] = False
            else:
                result['ai_report_available'] = False

            # Update statistics
            self.prediction_count += 1
            self.predictions_history.append({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'timestamp': result['timestamp']
            })

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def register_routes(self):
        """Register API routes"""

        @self.app.route('/')
        def index():
            """API home route"""
            return jsonify({
                'message': 'Brain Tumor Detection API with Gemini AI',
                'version': '2.0',
                'model_loaded': self.model is not None,
                'gemini_enabled': self.gemini_generator is not None,
                'endpoints': {
                    'health': '/api/health',
                    'predict': '/api/predict [POST]',
                    'predict_batch': '/api/predict/batch [POST]',
                    'classes': '/api/classes',
                    'stats': '/api/stats',
                    'model_info': '/api/model/info'
                }
            })

        @self.app.route('/api/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'gemini_enabled': self.gemini_generator is not None,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/classes')
        def get_classes():
            """Get available classification classes"""
            return jsonify({
                'classes': self.class_names,
                'num_classes': len(self.class_names)
            })

        @self.app.route('/api/model/info')
        def model_info():
            """Get model information"""
            if self.model is None:
                return jsonify({
                    'error': 'Model not loaded',
                    'message': 'Please train the model first'
                }), 503

            return jsonify({
                'model_loaded': True,
                'architecture': self.config['model']['architecture'],
                'input_shape': self.config['model']['input_shape'],
                'num_classes': len(self.class_names),
                'classes': self.class_names
            })

        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            """Predict tumor type from uploaded image"""
            if self.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded. Please train the model first.'
                }), 503

            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400

            file = request.files['file']

            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            if not self.allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'
                }), 400

            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(self.upload_folder, unique_filename)
                file.save(filepath)

                # Make prediction
                result = self.predict_image(filepath)

                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

                if result['success']:
                    return jsonify(result), 200
                else:
                    return jsonify(result), 500

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/predict/batch', methods=['POST'])
        def predict_batch():
            """Predict multiple images"""
            if self.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded'
                }), 503

            if 'files' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No files provided'
                }), 400

            files = request.files.getlist('files')

            if len(files) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No files selected'
                }), 400

            results = []

            for file in files:
                if file.filename == '' or not self.allowed_file(file.filename):
                    continue

                try:
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    filepath = os.path.join(self.upload_folder, unique_filename)
                    file.save(filepath)

                    result = self.predict_image(filepath)
                    result['filename'] = filename

                    results.append(result)

                    if os.path.exists(filepath):
                        os.remove(filepath)

                except Exception as e:
                    results.append({
                        'success': False,
                        'filename': file.filename,
                        'error': str(e)
                    })

            return jsonify({
                'success': True,
                'results': results,
                'total_processed': len(results)
            }), 200

        @self.app.route('/api/stats')
        def get_stats():
            """Get prediction statistics"""
            class_distribution = {}
            for pred in self.predictions_history:
                cls = pred['predicted_class']
                class_distribution[cls] = class_distribution.get(cls, 0) + 1

            return jsonify({
                'total_predictions': self.prediction_count,
                'class_distribution': class_distribution,
                'recent_predictions': self.predictions_history[-10:][::-1]  # Last 10
            })

    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask API"""
        print("\n" + "="*70)
        print("BRAIN TUMOR DETECTION API - STARTING")
        print("="*70)
        print(f"Model loaded: {self.model is not None}")
        print(f"Gemini AI enabled: {self.gemini_generator is not None}")
        print(f"Server: http://{host}:{port}")
        print("="*70 + "\n")

        self.app.run(host=host, port=port, debug=debug)


# Create API instance
if __name__ == '__main__':
    api = BrainTumorAPI()
    api.run(host='0.0.0.0', port=5000, debug=False)
