"""
Prediction utility for making predictions on new images
"""

import os
import cv2
import numpy as np
import yaml
from tensorflow import keras
from data_preprocessing import DataPreprocessor
import json


class BrainTumorPredictor:
    """
    Utility class for making predictions on brain tumor images
    """
    
    def __init__(self, model_path=None, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = DataPreprocessor(config_path)
        self.classes = self.config['classes']
        
        # Load model
        if model_path is None:
            model_path = f"{self.config['paths']['model_save_path']}/best_model.h5"
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def predict_single_image(self, image_path, return_all_probs=True):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            return_all_probs: Whether to return all class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        img = self.preprocessor.load_and_preprocess_image(image_path)
        img = self.preprocessor.apply_preprocessing(img)
        
        # Expand dimensions for batch
        img_array = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        class_probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(class_probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(class_probabilities[predicted_class_idx])
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_index': int(predicted_class_idx),
            'confidence': round(confidence * 100, 2),
            'image_path': image_path
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                self.classes[i]: round(float(class_probabilities[i]) * 100, 2)
                for i in range(len(self.classes))
            }
            
            # Get top 3 predictions
            top_k = min(3, len(self.classes))
            top_indices = np.argsort(class_probabilities)[-top_k:][::-1]
            
            result['top_predictions'] = [
                {
                    'class': self.classes[idx],
                    'confidence': round(float(class_probabilities[idx]) * 100, 2)
                }
                for idx in top_indices
            ]
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.predict_single_image(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'image_path': img_path
                })
        
        return results
    
    def predict_from_directory(self, directory_path, save_results=True):
        """
        Make predictions on all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            save_results: Whether to save results to JSON file
            
        Returns:
            Dictionary containing all prediction results
        """
        from pathlib import Path
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(directory_path).glob(f'*{ext}')))
            image_paths.extend(list(Path(directory_path).glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        # Make predictions
        results = self.predict_batch([str(p) for p in image_paths])
        
        # Organize results
        organized_results = {
            'total_images': len(results),
            'successful_predictions': sum(1 for r in results if 'error' not in r),
            'failed_predictions': sum(1 for r in results if 'error' in r),
            'class_distribution': {},
            'predictions': results
        }
        
        # Count class distribution
        for result in results:
            if 'predicted_class' in result:
                class_name = result['predicted_class']
                organized_results['class_distribution'][class_name] = \
                    organized_results['class_distribution'].get(class_name, 0) + 1
        
        # Save results
        if save_results:
            results_path = f"{self.config['paths']['results_path']}/batch_predictions.json"
            os.makedirs(self.config['paths']['results_path'], exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(organized_results, f, indent=4)
            
            print(f"Results saved to {results_path}")
        
        return organized_results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and results
        
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Make prediction
        result = self.predict_single_image(image_path)
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        ax1.imshow(img)
        ax1.set_title(f"Input MRI Image\n{os.path.basename(image_path)}", fontsize=12)
        ax1.axis('off')
        
        # Display predictions
        classes = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        
        colors = ['#667eea' if c == result['predicted_class'] else '#cccccc' for c in classes]
        
        ax2.barh(classes, probabilities, color=colors)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title(
            f"Predictions\nPredicted: {result['predicted_class'].upper()}\n"
            f"Confidence: {result['confidence']:.2f}%",
            fontsize=12,
            fontweight='bold'
        )
        ax2.set_xlim([0, 100])
        
        # Add percentage labels
        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            ax2.text(prob + 2, i, f"{prob:.1f}%", va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Demo prediction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Prediction')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, help='Path to model file', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BrainTumorPredictor(model_path=args.model)
    
    if args.image:
        # Single image prediction
        print(f"\nMaking prediction on: {args.image}")
        result = predictor.predict_single_image(args.image)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.2f}%")
        
        if args.visualize:
            predictor.visualize_prediction(args.image)
    
    elif args.directory:
        # Directory prediction
        print(f"\nMaking predictions on directory: {args.directory}")
        results = predictor.predict_from_directory(args.directory)
        
        print("\n" + "="*50)
        print("BATCH PREDICTION RESULTS")
        print("="*50)
        print(f"Total Images: {results['total_images']}")
        print(f"Successful: {results['successful_predictions']}")
        print(f"Failed: {results['failed_predictions']}")
        print("\nClass Distribution:")
        for class_name, count in results['class_distribution'].items():
            print(f"  {class_name}: {count}")
    
    else:
        print("Please provide either --image or --directory argument")
        parser.print_help()


if __name__ == "__main__":
    main()
