import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  shutil, random
import numpy as np


                      
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#split the data

raw_dir = "data/raw"
train_dir = "data/train"
test_dir  = "data/test"
val_dir   = "data/validation"



for folder in [train_dir, val_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

for cls in classes:
    class_path = os.path.join(raw_dir, cls)

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg','.png','.jpeg'))]

    if len(images) == 0:
        print(f"⚠ Warning: No images found in class '{cls}'")
        continue

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]


    for dataset, folder in zip(
        [train_images, val_images, test_images],
        [train_dir, val_dir, test_dir]
    ):
        class_folder = os.path.join(folder, cls)
        os.makedirs(class_folder, exist_ok=True)

        for img in dataset:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(class_folder, img)
            )

print("\Done! Data  split into Train / Test / Validation.")

#  Data processing(Augmentation) 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# DataLoader
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Number of classes =", num_classes)


print("Classes mapping:", train_generator.class_indices)
print("Number of training images:", train_generator.samples)
print("Number of validation images:", val_generator.samples)
print("Number of test images:", test_generator.samples)
