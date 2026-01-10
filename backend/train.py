"""
Training script for Brain Tumor Detection Model
Implements transfer learning with fine-tuning for high accuracy
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from model import BrainTumorModel
from data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    Handles the complete training pipeline for brain tumor detection
    """
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_builder = BrainTumorModel(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.history = None
        self.model = None
        
        # Create necessary directories
        os.makedirs(self.config['paths']['model_save_path'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_path'], exist_ok=True)
        os.makedirs(self.config['paths']['results_path'], exist_ok=True)
        os.makedirs(self.config['paths']['checkpoints_path'], exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data"""
        processed_path = f"{self.config['data']['processed_data_path']}/processed_data.npz"
        
        if not os.path.exists(processed_path):
            print("Processed data not found. Please run data_preprocessing.py first.")
            return None
        
        data = self.preprocessor.load_processed_data(processed_path)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        print(f"Data loaded successfully!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        return True
    
    def train_phase_1(self):
        """
        Phase 1: Train only the custom head with frozen base model
        """
        print("\n" + "="*70)
        print("TRAINING PHASE 1: Training custom classification head")
        print("="*70)
        
        # Build and compile model
        self.model = self.model_builder.build_model()
        self.model_builder.compile_model()
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks(
            model_path=f"{self.config['paths']['checkpoints_path']}/phase1_best_model.h5"
        )
        
        # Train
        history_phase1 = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=15,
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history_phase1
    
    def train_phase_2(self):
        """
        Phase 2: Fine-tune the entire model with unfrozen base layers
        """
        print("\n" + "="*70)
        print("TRAINING PHASE 2: Fine-tuning entire model")
        print("="*70)
        
        # Unfreeze base model
        self.model_builder.unfreeze_base_model(layers_to_unfreeze=-30)
        
        # Recompile with lower learning rate
        self.model_builder.compile_model(learning_rate=1e-5)
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks(
            model_path=f"{self.config['paths']['model_save_path']}/best_model.h5"
        )
        
        # Continue training
        history_phase2 = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history_phase2
    
    def train_full_pipeline(self):
        """Execute complete training pipeline"""
        print("\n" + "="*70)
        print("BRAIN TUMOR DETECTION MODEL - TRAINING PIPELINE")
        print("="*70)
        
        # Load data
        if not self.load_data():
            return
        
        # Phase 1: Train head
        history1 = self.train_phase_1()
        
        # Phase 2: Fine-tune
        history2 = self.train_phase_2()
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        }
        
        # Save training history
        with open(f"{self.config['paths']['results_path']}/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Load best model
        best_model_path = f"{self.config['paths']['model_save_path']}/best_model.h5"
        self.model = keras.models.load_model(best_model_path)
        
        # Evaluate
        test_results = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        
        print("\nTest Results:")
        print(f"Loss: {test_results[0]:.4f}")
        print(f"Accuracy: {test_results[1]:.4f}")
        print(f"Precision: {test_results[2]:.4f}")
        print(f"Recall: {test_results[3]:.4f}")
        print(f"AUC: {test_results[4]:.4f}")
        
        # Generate predictions
        y_pred_probs = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.config['classes']
        ))
        
        # Save results
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'test_auc': float(test_results[4]),
            'classification_report': classification_report(
                y_true_classes,
                y_pred_classes,
                target_names=self.config['classes'],
                output_dict=True
            )
        }
        
        with open(f"{self.config['paths']['results_path']}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return y_true_classes, y_pred_classes, y_pred_probs
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            history_path = f"{self.config['paths']['results_path']}/training_history.json"
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['paths']['results_path']}/training_history.png", dpi=300)
        print(f"Training history plot saved!")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config['classes'],
            yticklabels=self.config['classes'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.config['paths']['results_path']}/confusion_matrix.png", dpi=300)
        print(f"Confusion matrix saved!")
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_probs):
        """Plot ROC curves for each class"""
        n_classes = len(self.config['classes'])
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, color in enumerate(colors):
            plt.plot(
                fpr[i], tpr[i],
                color=color,
                lw=2,
                label=f'{self.config["classes"][i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config['paths']['results_path']}/roc_curves.png", dpi=300)
        print(f"ROC curves saved!")
        plt.close()
    
    def visualize_predictions(self, num_samples=16):
        """Visualize model predictions"""
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        predictions = self.model.predict(self.X_test[indices])
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test[indices], axis=1)
        
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img = self.X_test[indices[idx]]
                true_label = self.config['classes'][true_classes[idx]]
                pred_label = self.config['classes'][pred_classes[idx]]
                confidence = predictions[idx][pred_classes[idx]] * 100
                
                ax.imshow(img.astype('uint8'))
                
                color = 'green' if true_classes[idx] == pred_classes[idx] else 'red'
                ax.set_title(
                    f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
                    color=color,
                    fontweight='bold'
                )
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.config['paths']['results_path']}/predictions_visualization.png", dpi=300)
        print(f"Predictions visualization saved!")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("\nGenerating visualizations...")
        
        # Evaluate and get predictions
        y_true, y_pred, y_pred_probs = self.evaluate_model()
        
        # Generate plots
        self.plot_training_history()
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curves(y_true, y_pred_probs)
        self.visualize_predictions()
        
        print("\nAll visualizations generated successfully!")


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("ADVANCED BRAIN TUMOR DETECTION SYSTEM")
    print("="*70)
    
    trainer = ModelTrainer()
    
    # Train model
    trainer.train_full_pipeline()
    
    # Generate visualizations and evaluation
    trainer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel saved at: {trainer.config['paths']['model_save_path']}/best_model.h5")
    print(f"Results saved at: {trainer.config['paths']['results_path']}/")


if __name__ == "__main__":
    main()
