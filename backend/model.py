"""
Advanced Brain Tumor Detection Model
Uses EfficientNet-B4 with custom layers for high accuracy classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4, ResNet50V2, InceptionV3
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import yaml


class BrainTumorModel:
    """
    Advanced CNN model for brain tumor classification with multiple architecture options
    """
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_shape = tuple(self.config['model']['input_shape'])
        self.num_classes = self.config['model']['num_classes']
        self.architecture = self.config['model']['architecture']
        self.model = None
        
    def build_efficientnet_model(self):
        """Build model using EfficientNet-B4 backbone"""
        base_model = EfficientNetB4(
            include_top=False,
            weights='imagenet' if self.config['model']['pretrained'] else None,
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom head
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification head
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'] * 0.5)(x)
        
        x = layers.Dense(128, activation='relu', name='dense_3')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output'
        )(x)
        
        model = keras.Model(inputs, outputs, name='EfficientNetB4_BrainTumor')
        return model, base_model
    
    def build_resnet_model(self):
        """Build model using ResNet50V2 backbone"""
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet' if self.config['model']['pretrained'] else None,
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Rescaling(1./255)(inputs)
        x = base_model(x, training=False)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'] * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='ResNet50V2_BrainTumor')
        return model, base_model
    
    def build_inception_model(self):
        """Build model using InceptionV3 backbone"""
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet' if self.config['model']['pretrained'] else None,
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Rescaling(1./255)(inputs)
        x = base_model(x, training=False)
        
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'])(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['model']['dropout_rate'] * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='InceptionV3_BrainTumor')
        return model, base_model
    
    def build_model(self):
        """Build the model based on configuration"""
        if 'efficientnet' in self.architecture.lower():
            self.model, self.base_model = self.build_efficientnet_model()
        elif 'resnet' in self.architecture.lower():
            self.model, self.base_model = self.build_resnet_model()
        elif 'inception' in self.architecture.lower():
            self.model, self.base_model = self.build_inception_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        return self.model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with specified optimizer and loss"""
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
        
        optimizer_name = self.config['training']['optimizer'].lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.config['training']['loss'],
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
            ]
        )
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=self.config['paths']['logs_path'],
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def unfreeze_base_model(self, layers_to_unfreeze=-30):
        """Unfreeze base model layers for fine-tuning"""
        self.base_model.trainable = True
        
        # Freeze all layers except the last N
        for layer in self.base_model.layers[:layers_to_unfreeze]:
            layer.trainable = False
        
        print(f"Unfroze last {abs(layers_to_unfreeze)} layers of base model")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        return "Model not built yet"
    
    def count_parameters(self):
        """Count trainable and non-trainable parameters"""
        if self.model:
            trainable_count = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            non_trainable_count = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
            
            return {
                'trainable': trainable_count,
                'non_trainable': non_trainable_count,
                'total': trainable_count + non_trainable_count
            }
        return None


if __name__ == "__main__":
    # Test model creation
    model_builder = BrainTumorModel()
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    model_builder.get_model_summary()
    
    params = model_builder.count_parameters()
    print("\n" + "="*50)
    print("PARAMETER COUNT")
    print("="*50)
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Total parameters: {params['total']:,}")
