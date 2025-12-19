import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# Settings
# Determine paths based on file location
if os.path.exists("../data/test"):
    test_dir = "../data/test"
    train_dir = "../data/train"
    val_dir = "../data/validation"
    models_dir = "../models"
else:
    test_dir = "data/test"
    train_dir = "data/train"
    val_dir = "data/validation"
    models_dir = "models"

image_size = (224, 224)
batch_size = 16

# Load test data
print("=" * 60)
print("Loading test data...")
print("=" * 60)

# Load test data without shuffle to get correct labels
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Very important to get correct labels
)

# Get true labels
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Number of test images: {len(true_labels)}")
print(f"Class names: {class_names}")
print()

# Load VGG model
print("=" * 60)
print("Evaluating VGG16 Model...")
print("=" * 60)

vgg_path = os.path.join(models_dir, "model_vgg.keras")
if not os.path.exists(vgg_path):
    raise FileNotFoundError(f"Model file not found: {vgg_path}")

# Load model
model = tf.keras.models.load_model(vgg_path)
print(f"✓ Loaded VGG16 model")

# Prepare test data with VGG preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(test_gen, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Print results
print("\n" + "=" * 60)
print("VGG16 Model Evaluation Results:")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("=" * 60)

# Plot Confusion Matrix
print("\n" + "=" * 60)
print("Plotting Confusion Matrix...")
print("=" * 60)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - VGG16', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

confusion_matrix_path = os.path.join(models_dir, 'vgg_confusion_matrix.png')
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Confusion Matrix to: {confusion_matrix_path}")
plt.show()

# Plot metrics bar chart
print("\n" + "=" * 60)
print("Plotting Metrics Visualization...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add values on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.4f}\n({val*100:.2f}%)',
           ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('VGG16 Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')

plt.tight_layout()
metrics_path = os.path.join(models_dir, 'vgg_metrics.png')
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Metrics Visualization to: {metrics_path}")
plt.show()

# Comparison with other models (if they exist)
print("\n" + "=" * 60)
print("Model Comparison (if other models exist)...")
print("=" * 60)

comparison_data = {
    'Model': ['VGG16'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1]
}

# Check if other models exist and load their results
efficientnet_path = os.path.join(models_dir, "model_efficientnet.keras")
resnet_path = os.path.join(models_dir, "model_resnet.keras")

# Try to load EfficientNet if exists
if os.path.exists(efficientnet_path):
    try:
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
        eff_model = tf.keras.models.load_model(efficientnet_path)
        test_datagen_eff = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
        test_gen_eff = test_datagen_eff.flow_from_directory(
            test_dir, target_size=image_size, batch_size=batch_size,
            class_mode='categorical', shuffle=False
        )
        eff_predictions = eff_model.predict(test_gen_eff, verbose=0)
        eff_predicted_labels = np.argmax(eff_predictions, axis=1)
        eff_accuracy = accuracy_score(true_labels, eff_predicted_labels)
        eff_precision = precision_score(true_labels, eff_predicted_labels, average='weighted', zero_division=0)
        eff_recall = recall_score(true_labels, eff_predicted_labels, average='weighted', zero_division=0)
        eff_f1 = f1_score(true_labels, eff_predicted_labels, average='weighted', zero_division=0)
        
        comparison_data['Model'].append('EfficientNetB0')
        comparison_data['Accuracy'].append(eff_accuracy)
        comparison_data['Precision'].append(eff_precision)
        comparison_data['Recall'].append(eff_recall)
        comparison_data['F1-Score'].append(eff_f1)
        print("✓ Loaded EfficientNetB0 for comparison")
    except Exception as e:
        print(f"⚠ Could not load EfficientNetB0: {e}")

# Try to load ResNet if exists
if os.path.exists(resnet_path):
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        res_model = tf.keras.models.load_model(resnet_path)
        test_datagen_res = ImageDataGenerator(preprocessing_function=resnet_preprocess)
        test_gen_res = test_datagen_res.flow_from_directory(
            test_dir, target_size=image_size, batch_size=batch_size,
            class_mode='categorical', shuffle=False
        )
        res_predictions = res_model.predict(test_gen_res, verbose=0)
        res_predicted_labels = np.argmax(res_predictions, axis=1)
        res_accuracy = accuracy_score(true_labels, res_predicted_labels)
        res_precision = precision_score(true_labels, res_predicted_labels, average='weighted', zero_division=0)
        res_recall = recall_score(true_labels, res_predicted_labels, average='weighted', zero_division=0)
        res_f1 = f1_score(true_labels, res_predicted_labels, average='weighted', zero_division=0)
        
        comparison_data['Model'].append('ResNet50')
        comparison_data['Accuracy'].append(res_accuracy)
        comparison_data['Precision'].append(res_precision)
        comparison_data['Recall'].append(res_recall)
        comparison_data['F1-Score'].append(res_f1)
        print("✓ Loaded ResNet50 for comparison")
    except Exception as e:
        print(f"⚠ Could not load ResNet50: {e}")

# Print comparison table
if len(comparison_data['Model']) > 1:
    print("\n" + "-" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for i, model_name in enumerate(comparison_data['Model']):
        print(f"{model_name:<20} {comparison_data['Accuracy'][i]:<12.4f} "
              f"{comparison_data['Precision'][i]:<12.4f} {comparison_data['Recall'][i]:<12.4f} "
              f"{comparison_data['F1-Score'][i]:<12.4f}")
    print("-" * 60)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors_list = [['#3498db', '#e74c3c', '#2ecc71'], 
                   ['#e74c3c', '#3498db', '#2ecc71'],
                   ['#2ecc71', '#3498db', '#e74c3c'],
                   ['#f39c12', '#3498db', '#e74c3c']]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [comparison_data[metric][i] for i in range(len(comparison_data['Model']))]
        colors = colors_list[idx][:len(values)]
        bars = ax.bar(comparison_data['Model'], values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}\n({val*100:.2f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    comparison_path = os.path.join(models_dir, 'vgg_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {comparison_path}")
    plt.show()
    
    # Determine best model
    best_accuracy_idx = np.argmax(comparison_data['Accuracy'])
    best_f1_idx = np.argmax(comparison_data['F1-Score'])
    
    print("\n" + "=" * 60)
    print("Best Models:")
    print("=" * 60)
    print(f"Best Accuracy:  {comparison_data['Model'][best_accuracy_idx]} ({comparison_data['Accuracy'][best_accuracy_idx]:.4f})")
    print(f"Best F1-Score:  {comparison_data['Model'][best_f1_idx]} ({comparison_data['F1-Score'][best_f1_idx]:.4f})")
    print("=" * 60)
else:
    print("Only VGG16 model available. No comparison possible.")

# Print Classification Report
print("\n" + "=" * 60)
print("Classification Report:")
print("=" * 60)
report = classification_report(true_labels, predicted_labels, 
                               target_names=class_names, zero_division=0)
print(report)

# Save results to text file
results_path = os.path.join(models_dir, 'vgg_evaluation_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("VGG16 Model Evaluation Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Classification Report:\n")
    f.write("=" * 60 + "\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
    f.write("Confusion Matrix:\n")
    f.write("=" * 60 + "\n")
    f.write(f"Classes: {class_names}\n\n")
    f.write(str(cm))
    
    if len(comparison_data['Model']) > 1:
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Model Comparison:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        for i, model_name in enumerate(comparison_data['Model']):
            f.write(f"{model_name:<20} {comparison_data['Accuracy'][i]:<12.4f} "
                   f"{comparison_data['Precision'][i]:<12.4f} {comparison_data['Recall'][i]:<12.4f} "
                   f"{comparison_data['F1-Score'][i]:<12.4f}\n")

print(f"\n✓ Saved detailed results to: {results_path}")

# Check for Overfitting
print("\n" + "=" * 60)
print("Checking for Overfitting...")
print("=" * 60)

# Evaluate on training, validation, and test sets
print("\nEvaluating on different datasets...")

# Training set
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
train_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Evaluating on training set...")
train_predictions = model.predict(train_gen, verbose=0)
train_predicted_labels = np.argmax(train_predictions, axis=1)
train_true_labels = train_gen.classes
train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)

# Validation set
val_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Evaluating on validation set...")
val_predictions = model.predict(val_gen, verbose=0)
val_predicted_labels = np.argmax(val_predictions, axis=1)
val_true_labels = val_gen.classes
val_accuracy = accuracy_score(val_true_labels, val_predicted_labels)

# Test set (already calculated)
test_accuracy = accuracy

# Print comparison
print("\n" + "=" * 60)
print("Overfitting Analysis:")
print("=" * 60)
print(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("-" * 60)

train_val_gap = train_accuracy - val_accuracy
val_test_gap = val_accuracy - test_accuracy
train_test_gap = train_accuracy - test_accuracy

print(f"Train-Val Gap:       {train_val_gap:.4f} ({train_val_gap*100:.2f}%)")
print(f"Val-Test Gap:        {val_test_gap:.4f} ({val_test_gap*100:.2f}%)")
print(f"Train-Test Gap:      {train_test_gap:.4f} ({train_test_gap*100:.2f}%)")
print("=" * 60)

# Determine if overfitting exists
if train_val_gap > 0.10:  # More than 10% gap
    print("\n⚠ WARNING: Significant overfitting detected!")
    print(f"   Training accuracy is {train_val_gap*100:.2f}% higher than validation accuracy.")
    print("   The model is memorizing training data and not generalizing well.")
elif train_val_gap > 0.05:  # More than 5% gap
    print("\n⚠ CAUTION: Moderate overfitting detected!")
    print(f"   Training accuracy is {train_val_gap*100:.2f}% higher than validation accuracy.")
    print("   Consider using regularization techniques.")
else:
    print("\n✓ No significant overfitting detected!")
    print("   The model generalizes well to unseen data.")

# Plot overfitting visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart comparison
datasets = ['Training', 'Validation', 'Test']
accuracies = [train_accuracy, val_accuracy, test_accuracy]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax1.bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.4f}\n({val*100:.2f}%)',
           ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_title('Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Gap visualization
gaps = [train_val_gap, val_test_gap, train_test_gap]
gap_labels = ['Train-Val', 'Val-Test', 'Train-Test']
gap_colors = ['#e74c3c' if gap > 0.05 else '#2ecc71' for gap in gaps]

bars2 = ax2.bar(gap_labels, gaps, color=gap_colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars2, gaps):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.4f}\n({val*100:.2f}%)',
           ha='center', va='bottom' if val >= 0 else 'top', fontsize=11, fontweight='bold')

ax2.set_title('Accuracy Gaps (Overfitting Indicator)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy Gap', fontsize=12)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axhline(y=0.05, color='orange', linestyle='--', linewidth=1, label='Warning Threshold (5%)')
ax2.axhline(y=0.10, color='red', linestyle='--', linewidth=1, label='Critical Threshold (10%)')
ax2.legend()
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
overfitting_path = os.path.join(models_dir, 'vgg_overfitting_analysis.png')
plt.savefig(overfitting_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved overfitting analysis to: {overfitting_path}")
plt.show()

# Save overfitting analysis to file
with open(results_path, 'a', encoding='utf-8') as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("Overfitting Analysis:\n")
    f.write("=" * 60 + "\n")
    f.write(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
    f.write(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
    f.write(f"Train-Val Gap:       {train_val_gap:.4f} ({train_val_gap*100:.2f}%)\n")
    f.write(f"Val-Test Gap:        {val_test_gap:.4f} ({val_test_gap*100:.2f}%)\n")
    f.write(f"Train-Test Gap:      {train_test_gap:.4f} ({train_test_gap*100:.2f}%)\n")

print("\n" + "=" * 60)
print("Evaluation completed!")
print("=" * 60)