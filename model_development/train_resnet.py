import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import keras_tuner as kt


train_dir = "data/train"
val_dir = "data/validation"
test_dir = "data/test"

num_classes = 15
image_size = (224, 224)
batch_size = 16
epochs = 5


train_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,   
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
test_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_aug.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size,
    class_mode="categorical", shuffle=True
)

val_ds = val_aug.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size,
    class_mode="categorical", shuffle=False
)

test_ds = test_aug.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size,
    class_mode="categorical", shuffle=False
)


def build_model(hp):
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False   

    x = GlobalAveragePooling2D()(base.output)

    x = Dense(
        units=hp.Int('units', min_value=128, max_value=512, step=64),
        activation='relu'
    )(x)

    x = Dropout(
        rate=hp.Float('dropout', 0.2, 0.5, step=0.1)
    )(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)

    learning_rate = hp.Choice('lr', values=[1e-3, 1e-4, 5e-5])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='kt_tuner',
    project_name='resnet50_tuning'
)

tuner.search(train_ds, validation_data=val_ds, epochs=epochs)


print("\n Trials Results :\n")
for trial in tuner.oracle.get_best_trials(num_trials=3):
    print("Trial ID:", trial.trial_id)
    print("Hyperparameters:", trial.hyperparameters.values)
    print("Validation Accuracy:", trial.score, "\n")


best_model = tuner.get_best_models(num_models=1)[0]

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


os.makedirs("models", exist_ok=True)
best_model.save("models/model_resnet.keras")

print("\nTraining completed. Model saved in models folder.\n")


val_loss, val_acc = best_model.evaluate(val_ds)
test_loss, test_acc = best_model.evaluate(test_ds)

print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")


print("\n Best Model Summary:\n")
best_model.summary()

