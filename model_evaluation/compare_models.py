import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

np.random.seed(42)
tf.random.set_seed(42)

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(base_dir, 'data', 'test')
models_dir = os.path.join(base_dir, 'models')

image_size = (224, 224)
batch_size = 32

print("=" * 70)
print("MODEL EVALUATION - VGG16, EfficientNet, and ResNet")
print("=" * 70)


def evaluate_model(model_path, model_name, preprocess_func):
    """Evaluate a single model and return metrics"""
    print(f"\n[Evaluating {model_name}...]")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded {model_name}")

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        predictions = model.predict(test_gen, verbose=1)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = test_gen.classes

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None


def main():
    results = {}

    # VGG16
    vgg_result = evaluate_model(
        os.path.join(models_dir, 'model_vgg.keras'),
        'VGG16',
        vgg_preprocess
    )
    if vgg_result:
        results['vgg'] = vgg_result

    # EfficientNet
    efficientnet_result = evaluate_model(
        os.path.join(models_dir, 'model_efficientnet.keras'),
        'EfficientNetB0',
        efficientnet_preprocess
    )
    if efficientnet_result:
        results['efficientnet'] = efficientnet_result

    # ResNet
    resnet_result = evaluate_model(
        os.path.join(models_dir, 'model_resnet.keras'),
        'ResNet50',
        resnet_preprocess
    )
    if resnet_result:
        results['resnet'] = resnet_result

    # Save ONLY compare_models.txt
    compare_path = os.path.join(models_dir,'compare_models.txt')
    with open(compare_path, 'w') as f:
        f.write("MODEL COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 60 + "\n")

        for name, r in results.items():
            f.write(
                f"{r['model_name']:<15} "
                f"{r['accuracy']:<10.4f} {r['precision']:<10.4f} "
                f"{r['recall']:<10.4f} {r['f1_score']:<10.4f}\n"
            )

        f.write("=" * 60 + "\n")

    print("\nSaved: models/compare_models.txt")
    print("Done!")


if __name__ == "__main__":
    main()
