"""
Data preprocessing and augmentation pipeline for brain tumor detection
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import albumentations as A
from tqdm import tqdm
import json


class DataPreprocessor:
    """
    Handles data loading, preprocessing, and augmentation for brain tumor images
    """
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_shape = tuple(self.config['model']['input_shape'][:2])
        self.classes = self.config['classes']
        self.num_classes = len(self.classes)
        
    def load_and_preprocess_image(self, image_path, target_size=None):
        """Load and preprocess a single image"""
        if target_size is None:
            target_size = self.input_shape
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        return img
    
    def apply_preprocessing(self, img):
        """Apply advanced preprocessing techniques"""
        # Contrast Limited Adaptive Histogram Equalization
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return denoised
    
    def get_augmentation_pipeline(self, mode='train'):
        """Get albumentations augmentation pipeline"""
        if mode == 'train':
            transform = A.Compose([
                A.Rotate(
                    limit=self.config['augmentation']['rotation_range'],
                    p=0.7
                ),
                A.HorizontalFlip(p=0.5 if self.config['augmentation']['horizontal_flip'] else 0),
                A.ShiftScaleRotate(
                    shift_limit=self.config['augmentation']['width_shift_range'],
                    scale_limit=self.config['augmentation']['zoom_range'],
                    rotate_limit=0,
                    p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
                ], p=0.3),
            ])
        else:
            transform = A.Compose([])
        
        return transform
    
    def load_dataset_from_directory(self, data_dir, apply_preprocessing=True):
        """
        Load dataset from directory structure:
        data_dir/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                ...
        """
        data_dir = Path(data_dir)
        images = []
        labels = []
        
        print(f"Loading dataset from {data_dir}...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            print(f"Loading {len(image_files)} images from class '{class_name}'...")
            
            for img_path in tqdm(image_files, desc=class_name):
                try:
                    img = self.load_and_preprocess_image(img_path)
                    
                    if apply_preprocessing:
                        img = self.apply_preprocessing(img)
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"\nDataset loaded: {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def split_dataset(self, images, labels, test_size=0.1, val_size=0.2, random_state=42):
        """Split dataset into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_state
        )
        
        print("\nDataset split:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)
    
    def create_data_generators(self):
        """Create Keras ImageDataGenerator for training"""
        train_datagen = ImageDataGenerator(
            rotation_range=self.config['augmentation']['rotation_range'],
            width_shift_range=self.config['augmentation']['width_shift_range'],
            height_shift_range=self.config['augmentation']['height_shift_range'],
            horizontal_flip=self.config['augmentation']['horizontal_flip'],
            vertical_flip=self.config['augmentation']['vertical_flip'],
            zoom_range=self.config['augmentation']['zoom_range'],
            brightness_range=self.config['augmentation']['brightness_range'],
            fill_mode=self.config['augmentation']['fill_mode']
        )
        
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def save_processed_data(self, data, save_path):
        """Save processed data to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **data)
        print(f"Data saved to {save_path}")
    
    def load_processed_data(self, load_path):
        """Load processed data from disk"""
        data = np.load(load_path)
        return {key: data[key] for key in data.files}
    
    def get_class_weights(self, labels):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"\nClass weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def visualize_samples(self, images, labels, num_samples=16, save_path=None):
        """Visualize sample images from dataset"""
        import matplotlib.pyplot as plt
        
        num_samples = min(num_samples, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img_idx = indices[idx]
                img = images[img_idx]
                label = labels[img_idx] if labels.ndim == 1 else np.argmax(labels[img_idx])
                
                ax.imshow(img)
                ax.set_title(f"Class: {self.classes[label]}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def prepare_dataset(config_path='config.yaml'):
    """Main function to prepare dataset"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(config_path)
    
    # Load dataset
    raw_data_path = config['data']['raw_data_path']
    
    if not os.path.exists(raw_data_path):
        print(f"Creating data directory structure...")
        for split in ['train', 'val', 'test']:
            for class_name in config['classes']:
                os.makedirs(f"{config['data']['processed_data_path']}/{split}/{class_name}", exist_ok=True)
        
        print("\nPlease add your dataset to the following directory structure:")
        print(f"{raw_data_path}/")
        for class_name in config['classes']:
            print(f"    {class_name}/")
            print(f"        image1.jpg")
            print(f"        image2.jpg")
            print(f"        ...")
        return
    
    # Load and preprocess
    images, labels = preprocessor.load_dataset_from_directory(raw_data_path)
    
    # Split dataset
    train_data, val_data, test_data = preprocessor.split_dataset(images, labels)
    
    # Save processed data
    processed_path = config['data']['processed_data_path']
    preprocessor.save_processed_data(
        {
            'X_train': train_data[0], 'y_train': train_data[1],
            'X_val': val_data[0], 'y_val': val_data[1],
            'X_test': test_data[0], 'y_test': test_data[1]
        },
        f"{processed_path}/processed_data.npz"
    )
    
    # Visualize samples
    preprocessor.visualize_samples(
        train_data[0][:16],
        train_data[1][:16],
        save_path='results/sample_visualization.png'
    )
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    prepare_dataset()
