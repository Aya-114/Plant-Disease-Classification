import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
test_dir = os.path.join(base_dir, 'data', 'test')
models_dir = os.path.join(base_dir, 'models')

# Create output directory for visualizations
os.makedirs(os.path.join(models_dir, 'evaluation'), exist_ok=True)

image_size = (224, 224)
batch_size = 32

print("=" * 70)
print("MODEL EVALUATION - EfficientNet and ResNet")
print("=" * 70)

def evaluate_model(model_path, model_name, preprocess_func):
    """Evaluate a single model and return metrics"""
    print(f"\n[Evaluating {model_name}...]")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded {model_name} model")
        
        # Prepare test data
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Get class names
        class_names = list(test_gen.class_indices.keys())
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(test_gen, verbose=1)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = test_gen.classes
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Save classification report
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=class_names, zero_division=0)
        report_path = os.path.join(models_dir, 'evaluation', f'{model_name.lower()}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        # Print results
        print(f"\n{model_name} Results:")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = os.path.join(models_dir, 'evaluation', f'{model_name.lower()}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'class_report': report
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

def main():
    # Evaluate both models
    results = {}
    
    # Evaluate EfficientNet
    efficientnet_result = evaluate_model(
        os.path.join(models_dir, 'model_efficientnet.keras'),
        'EfficientNetB0',
        efficientnet_preprocess
    )
    if efficientnet_result:
        results['efficientnet'] = efficientnet_result
    
    # Evaluate ResNet
    resnet_result = evaluate_model(
        os.path.join(models_dir, 'model_resnet.keras'),
        'ResNet50',
        resnet_preprocess
    )
    if resnet_result:
        results['resnet'] = resnet_result
    
    # Print comparison
    if results:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        for name, result in results.items():
            print(f"{result['model_name']:<15} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1_score']:<10.4f}")
        print("=" * 70)
    
    print("\nEvaluation complete! Check the 'models/evaluation' directory for detailed reports.")

if _name_ == "_main_":
    main()