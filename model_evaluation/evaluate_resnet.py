import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def evaluate_model():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'data', 'test')
    model_path = os.path.join(base_dir, 'models', 'model_resnet.keras')
    output_dir = os.path.join(base_dir, 'models', 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    image_size = (224, 224)
    batch_size = 32
    model_name = 'ResNet50'

    print("=" * 70)
    print(f"{model_name} EVALUATION")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading {model_name} model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare test data
    print("\nLoading test data...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
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
    print("\nMaking predictions...")
    predictions = model.predict(test_gen, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_gen.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"{model_name} EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Save classification report
    report = classification_report(true_labels, predicted_labels, 
                                 target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, 'resnet_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'resnet_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("-" * 40)
    print(f"- Classification report: resnet_report.txt")
    print(f"- Confusion matrix: resnet_confusion_matrix.png")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_model()
