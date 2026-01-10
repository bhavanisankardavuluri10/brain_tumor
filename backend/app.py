"""
Flask API Backend for Brain Tumor Detection
Provides RESTful endpoints for image upload and prediction
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

from data_preprocessing import DataPreprocessor


class BrainTumorAPI:
    """
    API class for brain tumor detection
    """
    
    def __init__(self, config_path='config.yaml'):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.model = None
        self.preprocessor = DataPreprocessor(config_path)
        self.upload_folder = self.config['api']['upload_folder']
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Register routes
        self.register_routes()
        
        # Statistics
        self.prediction_count = 0
        self.predictions_history = []
    
    def load_model(self):
        """Load the trained model"""
        model_path = f"{self.config['paths']['model_save_path']}/best_model.h5"
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Please train the model first using train.py")
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        allowed_extensions = self.config['api']['allowed_extensions']
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load and preprocess
        img = self.preprocessor.load_and_preprocess_image(image_path)
        img = self.preprocessor.apply_preprocessing(img)
        
        # Expand dimensions for batch
        img_array = np.expand_dims(img, axis=0)
        
        return img_array, img
    
    def predict_image(self, image_path):
        """Make prediction on image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess
            img_array, original_img = self.preprocess_image(image_path)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            class_probabilities = predictions[0]
            
            # Get results
            predicted_class_idx = np.argmax(class_probabilities)
            predicted_class = self.config['classes'][predicted_class_idx]
            confidence = float(class_probabilities[predicted_class_idx]) * 100
            
            # Get top 3 predictions
            top_k = min(3, len(self.config['classes']))
            top_indices = np.argsort(class_probabilities)[-top_k:][::-1]
            
            top_predictions = [
                {
                    'class': self.config['classes'][idx],
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
                    self.config['classes'][i]: round(float(class_probabilities[i]) * 100, 2)
                    for i in range(len(self.config['classes']))
                },
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add warning for tumor cases
            if predicted_class != 'no_tumor' and is_confident:
                result['warning'] = f"Tumor detected: {predicted_class}. Please consult a medical professional."
            
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
                'message': 'Brain Tumor Detection API',
                'version': '1.0',
                'endpoints': {
                    'predict': '/api/predict',
                    'health': '/api/health',
                    'stats': '/api/stats',
                    'classes': '/api/classes'
                }
            })
        
        @self.app.route('/api/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/classes', methods=['GET'])
        def get_classes():
            """Get available classes"""
            return jsonify({
                'classes': self.config['classes'],
                'num_classes': len(self.config['classes'])
            })
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            # Check if model is loaded
            if self.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded. Please train the model first.'
                }), 500
            
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400
            
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Check file extension
            if not self.allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Invalid file type. Allowed types: {self.config["api"]["allowed_extensions"]}'
                }), 400
            
            try:
                # Save file
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(self.upload_folder, unique_filename)
                file.save(filepath)
                
                # Make prediction
                result = self.predict_image(filepath)
                
                # Update statistics
                if result.get('success', False):
                    self.prediction_count += 1
                    self.predictions_history.append({
                        'filename': filename,
                        'prediction': result['predicted_class'],
                        'confidence': result['confidence'],
                        'timestamp': result['timestamp']
                    })
                    
                    # Keep only last 100 predictions
                    if len(self.predictions_history) > 100:
                        self.predictions_history = self.predictions_history[-100:]
                
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/predict/batch', methods=['POST'])
        def predict_batch():
            """Batch prediction endpoint"""
            if self.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded'
                }), 500
            
            if 'files' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No files provided'
                }), 400
            
            files = request.files.getlist('files')
            results = []
            
            for file in files:
                if file and self.allowed_file(file.filename):
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
                            'filename': file.filename,
                            'success': False,
                            'error': str(e)
                        })
            
            return jsonify({
                'success': True,
                'results': results,
                'total_processed': len(results)
            })
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get API statistics"""
            class_distribution = {}
            for pred in self.predictions_history:
                class_name = pred['prediction']
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            
            return jsonify({
                'total_predictions': self.prediction_count,
                'recent_predictions': len(self.predictions_history),
                'class_distribution': class_distribution,
                'recent_history': self.predictions_history[-10:]
            })
        
        @self.app.route('/api/model/info', methods=['GET'])
        def get_model_info():
            """Get model information"""
            if self.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded'
                }), 500
            
            return jsonify({
                'model_name': self.config['model']['name'],
                'architecture': self.config['model']['architecture'],
                'input_shape': self.config['model']['input_shape'],
                'classes': self.config['classes'],
                'num_classes': self.config['model']['num_classes'],
                'confidence_threshold': self.config['prediction']['confidence_threshold']
            })
    
    def run(self):
        """Run the Flask application"""
        host = self.config['api']['host']
        port = self.config['api']['port']
        debug = self.config['api']['debug']
        
        print("\n" + "="*70)
        print("BRAIN TUMOR DETECTION API SERVER")
        print("="*70)
        print(f"Server running on http://{host}:{port}")
        print(f"API Documentation: http://{host}:{port}/")
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to start API"""
    api = BrainTumorAPI()
    api.run()


if __name__ == '__main__':
    main()
