import tensorflow as tf
import os
from gradcam_utils import make_gradcam_heatmap, save_gradcam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


base_dir = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(base_dir, "..", "models", "model_efficientnet.keras")
test_dir = os.path.join(base_dir, "..", "data", "test")
SAVE_ROOT = os.path.join(base_dir, "..", "models", "gradcam_efficientnet")


print("Loading EfficientNet model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
last_conv_layer = "top_conv"  


for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    
    save_folder = os.path.join(SAVE_ROOT, folder)
    os.makedirs(save_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        

        print(f"Generating Grad-CAM for {folder}/{file}...")
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

     
        save_path = os.path.join(save_folder, file)
        save_gradcam(image_path, heatmap, save_path)
        print("Saved:", save_path)

print("All Grad-CAM images generated successfully!")