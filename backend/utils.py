"""
Utility functions for the Brain Tumor Detection System
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml


def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw/glioma',
        'data/raw/meningioma',
        'data/raw/pituitary',
        'data/raw/no_tumor',
        'data/processed',
        'models',
        'logs',
        'results',
        'uploads',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def check_dataset(data_dir='data/raw'):
    """Check dataset statistics"""
    data_dir = Path(data_dir)
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    total_images = 0
    class_counts = {}
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.jpeg')) + \
                    list(class_dir.glob('*.png'))
            
            count = len(images)
            class_counts[class_dir.name] = count
            total_images += count
            
            print(f"\n{class_dir.name}:")
            print(f"  Images: {count}")
            
            if count > 0:
                # Sample image info
                sample_img = cv2.imread(str(images[0]))
                if sample_img is not None:
                    print(f"  Sample size: {sample_img.shape}")
    
    print(f"\nTotal Images: {total_images}")
    
    # Check balance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"\nDataset Balance Ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("⚠️  Warning: Dataset is imbalanced. Consider data augmentation.")
        else:
            print("✓ Dataset is reasonably balanced.")
    
    return class_counts, total_images


def visualize_class_distribution(class_counts, save_path='results/class_distribution.png'):
    """Visualize class distribution"""
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Tumor Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (cls, cnt) in enumerate(zip(classes, counts)):
        plt.text(i, cnt + max(counts)*0.02, str(cnt), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nClass distribution plot saved to: {save_path}")
    plt.close()


def test_model_inference_speed(model_path='models/best_model.h5', num_tests=10):
    """Test model inference speed"""
    import time
    from tensorflow import keras
    
    print("\n" + "="*50)
    print("MODEL INFERENCE SPEED TEST")
    print("="*50)
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create dummy input
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Warm-up
    model.predict(dummy_input, verbose=0)
    
    # Time predictions
    times = []
    for i in range(num_tests):
        start_time = time.time()
        model.predict(dummy_input, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nNumber of tests: {num_tests}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Standard deviation: {std_time*1000:.2f} ms")
    print(f"Min time: {min(times)*1000:.2f} ms")
    print(f"Max time: {max(times)*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} images/second")


def compare_preprocessing_methods(image_path):
    """Compare different preprocessing methods"""
    import cv2
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Original
    original = img.copy()
    
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Histogram Equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced)
    axes[0, 1].set_title('CLAHE Enhanced', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(denoised)
    axes[1, 0].set_title('CLAHE + Denoised', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hist_eq)
    axes[1, 1].set_title('Histogram Equalization', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    print("Preprocessing comparison saved to: results/preprocessing_comparison.png")
    plt.close()


def generate_model_report(config_path='config.yaml'):
    """Generate comprehensive model report"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load evaluation results
    results_path = f"{config['paths']['results_path']}/evaluation_results.json"
    
    if not os.path.exists(results_path):
        print("Evaluation results not found. Please train and evaluate the model first.")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Generate report
    report = f"""
{'='*70}
BRAIN TUMOR DETECTION SYSTEM - MODEL REPORT
{'='*70}

Model Configuration:
--------------------
Architecture: {config['model']['architecture']}
Input Shape: {config['model']['input_shape']}
Number of Classes: {config['model']['num_classes']}
Classes: {', '.join(config['classes'])}

Training Configuration:
-----------------------
Batch Size: {config['training']['batch_size']}
Epochs: {config['training']['epochs']}
Learning Rate: {config['training']['learning_rate']}
Optimizer: {config['training']['optimizer']}

Model Performance:
------------------
Test Accuracy: {results['test_accuracy']*100:.2f}%
Test Precision: {results['test_precision']*100:.2f}%
Test Recall: {results['test_recall']*100:.2f}%
Test AUC: {results['test_auc']*100:.2f}%

Per-Class Performance:
----------------------
"""
    
    # Add per-class metrics
    class_report = results['classification_report']
    for class_name in config['classes']:
        if class_name in class_report:
            metrics = class_report[class_name]
            report += f"\n{class_name.upper()}:\n"
            report += f"  Precision: {metrics['precision']*100:.2f}%\n"
            report += f"  Recall: {metrics['recall']*100:.2f}%\n"
            report += f"  F1-Score: {metrics['f1-score']*100:.2f}%\n"
            report += f"  Support: {metrics['support']}\n"
    
    report += f"\n{'='*70}\n"
    
    # Save report
    report_path = f"{config['paths']['results_path']}/model_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")


def main():
    """Main utility function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Utilities')
    parser.add_argument('--create-dirs', action='store_true', help='Create project directories')
    parser.add_argument('--check-dataset', action='store_true', help='Check dataset statistics')
    parser.add_argument('--test-speed', action='store_true', help='Test inference speed')
    parser.add_argument('--compare-preprocessing', type=str, help='Compare preprocessing methods')
    parser.add_argument('--generate-report', action='store_true', help='Generate model report')
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directories()
    
    if args.check_dataset:
        class_counts, total = check_dataset()
        visualize_class_distribution(class_counts)
    
    if args.test_speed:
        test_model_inference_speed()
    
    if args.compare_preprocessing:
        compare_preprocessing_methods(args.compare_preprocessing)
    
    if args.generate_report:
        generate_model_report()
    
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
