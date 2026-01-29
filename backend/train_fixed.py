"""
Brain Tumor Detection - Fixed Training Script
==============================================
This script fixes the overfitting problem where model predicts only one class.

Key Fixes:
1. Proper class weights for imbalanced data
2. Strong data augmentation
3. Gradual unfreezing (3-phase training)
4. Label smoothing to prevent overconfidence
5. Better regularization (L2 + Dropout)
"""
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportIndexIssue=false
# type: ignore

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from collections import Counter

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


class BrainTumorTrainer:
    """Fixed trainer for brain tumor classification."""
    
    def __init__(self):
        # Get base directory (where this script is located)
        self.base_dir = Path(__file__).parent.absolute()
        
        # Load config
        config_path = self.base_dir / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup GPU
        self._setup_gpu()
        
        # Create directories
        self._setup_directories()
        
        # Initialize variables
        self.model = None
        self.base_model = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.class_weights = None
        
        # Class names (must match folder names exactly)
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Timestamp for saving files
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _setup_gpu(self):
        """Configure GPU memory growth."""
        print("\n" + "="*60)
        print("GPU SETUP")
        print("="*60)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    print(f"  Warning: {e}")
        else:
            print("No GPU found - using CPU")
            
    def _setup_directories(self):
        """Create output directories."""
        dirs = [
            self.base_dir / 'models' / 'trained',
            self.base_dir / 'models' / 'checkpoints',
            self.base_dir / 'results' / 'metrics',
            self.base_dir / 'results' / 'plots',
            self.base_dir / 'logs'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def prepare_data(self):
        """Prepare data generators with augmentation."""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Data paths
        data_path = self.base_dir.parent / 'data' / 'raw'
        train_path = data_path / 'Training'
        test_path = data_path / 'Testing'
        
        print(f"Training data: {train_path}")
        print(f"Testing data: {test_path}")
        
        # Check class distribution
        print("\nClass distribution in training data:")
        for cls in self.class_names:
            cls_path = train_path / cls
            if cls_path.exists():
                count = len(list(cls_path.glob('*')))
                print(f"  {cls}: {count} images")
        
        # Image settings
        img_size = (224, 224)
        batch_size = 32
        
        # Strong augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # No augmentation for test
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # Create generators
        print("\nCreating data generators...")
        
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Update class names from generator
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"\nClasses detected: {self.class_names}")
        
        # Compute class weights
        self._compute_class_weights()
        
        print(f"\nData ready:")
        print(f"  Training samples: {self.train_generator.samples}")
        print(f"  Validation samples: {self.val_generator.samples}")
        print(f"  Test samples: {self.test_generator.samples}")
        
    def _compute_class_weights(self):
        """Compute class weights to handle imbalanced data."""
        print("\nComputing class weights...")
        
        # Get class distribution
        class_counts = Counter(self.train_generator.classes)
        total = sum(class_counts.values())
        n_classes = len(class_counts)
        
        # Compute weights: weight = total / (n_classes * count)
        self.class_weights = {}
        for cls_idx, count in class_counts.items():
            weight = total / (n_classes * count)
            self.class_weights[cls_idx] = weight
            cls_name = self.class_names[cls_idx]
            print(f"  {cls_name}: {count} samples, weight = {weight:.3f}")
            
    def build_model(self):
        """Build model with MobileNetV2 backbone (lighter and faster)."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        input_shape = (224, 224, 3)
        num_classes = 4
        
        # Load pretrained MobileNetV2 (lighter than EfficientNet)
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Freeze base model initially
        self.base_model.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=input_shape)
        
        # Base model
        x = self.base_model(inputs, training=False)
        
        # Classification head with regularization
        x = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        print(f"Model built successfully")
        print(f"Total parameters: {self.model.count_params():,}")
        
    def compile_model(self, learning_rate=0.001):
        """Compile model with label smoothing."""
        print(f"\nCompiling with learning rate: {learning_rate}")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
    def get_callbacks(self, phase):
        """Get training callbacks."""
        checkpoint_path = self.base_dir / 'models' / 'checkpoints' / f'best_{phase}_{self.timestamp}.keras'
        
        return [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
    def train_phase1(self, epochs=10):
        """Phase 1: Train only classification head (base frozen)."""
        print("\n" + "="*60)
        print("PHASE 1: FEATURE EXTRACTION")
        print("="*60)
        print("Training classification head with frozen base model...")
        
        self.base_model.trainable = False
        self.compile_model(learning_rate=0.001)
        
        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks('phase1'),
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history
        
    def train_phase2(self, epochs=15):
        """Phase 2: Fine-tune top layers of base model."""
        print("\n" + "="*60)
        print("PHASE 2: FINE-TUNING TOP LAYERS")
        print("="*60)
        
        # Unfreeze top 50 layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False
            
        trainable_count = sum(1 for layer in self.base_model.layers if layer.trainable)
        print(f"Unfroze {trainable_count} layers")
        
        self.compile_model(learning_rate=0.0001)
        
        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks('phase2'),
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history
        
    def train_phase3(self, epochs=10):
        """Phase 3: Full fine-tuning with very low learning rate."""
        print("\n" + "="*60)
        print("PHASE 3: FULL FINE-TUNING")
        print("="*60)
        
        # Unfreeze all layers
        self.base_model.trainable = True
        
        self.compile_model(learning_rate=0.00001)
        
        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks('phase3'),
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history
        
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Get predictions
        self.test_generator.reset()
        y_true = self.test_generator.classes
        
        print("Generating predictions...")
        y_pred_probs = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n{'='*40}")
        print("RESULTS")
        print(f"{'='*40}")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        
        # Check for single-class prediction (overfitting symptom)
        unique_preds = np.unique(y_pred)
        print(f"\nUnique classes predicted: {len(unique_preds)}/{len(self.class_names)}")
        print(f"Classes: {[self.class_names[i] for i in unique_preds]}")
        
        if len(unique_preds) == 1:
            print("\n⚠️ WARNING: Model is predicting only ONE class!")
            print("This indicates severe overfitting - model needs retraining.")
        elif len(unique_preds) < len(self.class_names):
            print("\n⚠️ WARNING: Model is not predicting all classes!")
        else:
            print("\n✓ Model is predicting all classes correctly!")
        
        # Print classification report
        print(f"\n{'='*40}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*40}")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classes_predicted': int(len(unique_preds)),
            'timestamp': self.timestamp
        }
        
        metrics_path = self.base_dir / 'results' / 'metrics' / f'metrics_{self.timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return y_true, y_pred, y_pred_probs
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plot_path = self.base_dir / 'results' / 'plots' / f'confusion_matrix_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")
        
    def plot_roc_curves(self, y_true, y_pred_probs):
        """Plot ROC curves."""
        print("Generating ROC curves...")
        
        y_true_bin = label_binarize(y_true, classes=list(range(len(self.class_names))))
        
        plt.figure(figsize=(10, 8))
        
        for i, cls_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{cls_name} (AUC={roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.base_dir / 'results' / 'plots' / f'roc_curves_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")
        
    def save_model(self):
        """Save trained model."""
        print("\nSaving model...")
        
        # Save with timestamp
        model_path = self.base_dir / 'models' / 'trained' / f'model_{self.timestamp}.keras'
        self.model.save(str(model_path))
        print(f"Saved: {model_path}")
        
        # Save as final model (overwrites previous)
        final_path = self.base_dir / 'models' / 'trained' / 'final_model.keras'
        self.model.save(str(final_path))
        print(f"Saved: {final_path}")
        
        # Save class info
        class_info = {
            'classes': self.class_names,
            'class_indices': self.train_generator.class_indices,
            'timestamp': self.timestamp
        }
        info_path = self.base_dir / 'models' / 'trained' / 'class_info.json'
        with open(info_path, 'w') as f:
            json.dump(class_info, f, indent=4)
        print(f"Saved: {info_path}")
        
    def run(self):
        """Run complete training pipeline."""
        print("\n" + "="*60)
        print("BRAIN TUMOR DETECTION - TRAINING PIPELINE")
        print("="*60)
        print("\nThis training script fixes overfitting with:")
        print("  • Class weights for imbalanced data")
        print("  • Strong data augmentation")
        print("  • 3-phase gradual unfreezing")
        print("  • Label smoothing (0.1)")
        print("  • L2 regularization + Dropout")
        
        start_time = time.time()
        
        try:
            # 1. Prepare data
            self.prepare_data()
            
            # 2. Build model
            self.build_model()
            
            # 3. Phase 1: Feature extraction
            self.train_phase1(epochs=10)
            
            # 4. Phase 2: Fine-tune top layers
            self.train_phase2(epochs=15)
            
            # 5. Phase 3: Full fine-tuning
            self.train_phase3(epochs=10)
            
            # 6. Evaluate
            y_true, y_pred, y_pred_probs = self.evaluate()
            
            # 7. Plot results
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_roc_curves(y_true, y_pred_probs)
            
            # 8. Save model
            self.save_model()
            
            total_time = (time.time() - start_time) / 60
            
            print("\n" + "="*60)
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total time: {total_time:.1f} minutes")
            print(f"\nModel saved to: backend/models/trained/final_model.keras")
            print("\nTo test the model, run the web app:")
            print("  python backend/app.py")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    trainer = BrainTumorTrainer()
    trainer.run()
