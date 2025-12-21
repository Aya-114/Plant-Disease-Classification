import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os


def predict(image, model):
    import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

import sys
import traceback

class ConsoleRedirect:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert("end", message)
        self.text_widget.see("end")

    def flush(self):
        pass


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")

IMAGE_SIZE = (224, 224)

MODEL_CONFIG = {
    "vgg": {
        "path": os.path.join(MODELS_DIR, "train_result_models" , "model_vgg.keras"),
        "preprocess": vgg_preprocess,
        "name": "VGG16"
    },
    "resnet": {
        "path": os.path.join(MODELS_DIR, "train_result_models", "model_resnet.keras"),
        "preprocess": resnet_preprocess,
        "name": "ResNet50"
    },
    "efficientnet": {
        "path": os.path.join(MODELS_DIR, "train_result_models", "model_efficientnet.keras"),
        "preprocess": efficientnet_preprocess,
        "name": "EfficientNetB0"
    }
}
MODELS = ["resnet", "vgg", "efficientnet"]
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(base_dir, 'data', 'test')
models_dir = os.path.join(base_dir, 'models')
LOADED_MODELS = {}

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))


def predict(image, model_name):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIG[model_name]

    
    if model_name not in LOADED_MODELS:
        LOADED_MODELS[model_name] = tf.keras.models.load_model(config["path"])

    model = LOADED_MODELS[model_name]

    
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)

    if img_array.shape[-1] == 4: 
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = config["preprocess"](img_array)

    
    preds = model.predict(img_array, verbose=0)[0]

    
    top3_idx = np.argsort(preds)[::-1][:3]

    results = [
        (CLASS_NAMES[i], float(preds[i]))
        for i in top3_idx
    ]

    return results


def load_gradcam_image(image_path, model_name):

    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)

    gradcam_dir = {
        "vgg": "gradcam_vgg",
        "resnet": "gradcam_resnet",
        "efficientnet": "gradcam_efficientnet"
    }

    base_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "gradcam"
    )

    search_dir = os.path.join(base_models_dir, gradcam_dir[model_name])

    for root, dirs, files in os.walk(search_dir):
        for file in files:
            file_name, ext = os.path.splitext(file)
            if file_name.lower() == name.lower():
                gradcam_path = os.path.join(root, file)
                return Image.open(gradcam_path)

    raise FileNotFoundError(f"Grad-CAM not found recursively for '{name}' in {search_dir}")



class App:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Image Classification GUI")
        self.root.geometry("1000x900")

        self.image = None
        self.tk_img = None

        
        left = tk.Frame(root)
        left.pack(side="left", padx=10, pady=10)

        self.image_label = tk.Label(left, text="No Image")
        self.image_label.pack()
        
        console_frame = tk.Frame(root)
        console_frame.pack(fill="both", padx=10, pady=5)
        

        tk.Label(console_frame, text="Console Output", font=("Arial", 11, "bold")).pack(anchor="w")

        self.console = tk.Text(
            console_frame,
            height=20,
            width=50,
            bg="black",
            fg="lime",
            font=("Courier", 9)
        )
        self.console.pack(fill="both", expand=True)
        sys.stdout = ConsoleRedirect(self.console)
        sys.stderr = ConsoleRedirect(self.console)
        scrollbar = tk.Scrollbar(self.console)
        scrollbar.pack(side="right", fill="y")
        self.console.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.console.yview)


        tk.Button(left, text="Upload Image", command=self.upload_image).pack(pady=5)
        
        tk.Label(left, text="Select Model").pack(pady=5)
        self.model_var = tk.StringVar(value=MODELS[0])
        ttk.Combobox(left, textvariable=self.model_var, values=MODELS).pack()

        tk.Button(left, text="Run Inference", command=self.run_inference).pack(pady=10)
        tk.Button(left, text="Generate Grad-CAM", command=self.run_gradcam).pack(pady=5)

        
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="Top-3 Predictions", font=("Arial", 12, "bold")).pack()

        self.result_text = tk.Text(right, height=8, width=50)
        self.result_text.pack(pady=5)

        tk.Label(right, text="Grad-CAM Visualization", font=("Arial", 12, "bold")).pack()
        self.gradcam_label = tk.Label(right)
        self.gradcam_label.pack(pady=5)

        tk.Button(left, text="Compare Models", command=self.run_compare).pack(pady=10)
        tk.Label(right, text="Comparison Output", font=("Arial", 12, "bold")).pack()
        self.compare_text = tk.Text(right, height=12)
        self.compare_text.pack()
        
    
    def upload_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.image_path = path
        self.image = Image.open(path)
        self.show_image()

    def show_image(self):
        self.tk_img = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.tk_img, text="")

    
    def run_inference(self):
        if self.image is None:
            return

        model = self.model_var.get()
        results = predict(self.image, model)

        self.result_text.delete("1.0", tk.END)
        for cls, score in results:
            self.result_text.insert(tk.END, f"{cls}: {score:.3f}\n")

    
    def run_gradcam(self):
        if self.image is None:
            print("No image selected")
            return

        if not self.image_path:
            print("No image path to load Grad-CAM from")
          

        try:
            model = self.model_var.get()
            cam_img = load_gradcam_image(self.image_path, model)
            cam_img = cam_img.resize((300, 300))
            self.gradcam_img = ImageTk.PhotoImage(cam_img)
            self.gradcam_label.config(image=self.gradcam_img)

        except Exception as e:
            print("Grad-CAM loading error:")
            import traceback
            traceback.print_exc()

    def run_compare(self):
        if self.image is None:
            print("No image selected")
            return

        results = self.compare_models(self.image)

        self.compare_text.delete("1.0", tk.END)
        for model, preds in results.items():
            self.compare_text.insert(tk.END, f"Model: {model}\n")
            for cls, conf in preds:
                self.compare_text.insert(tk.END, f"  {cls}: {conf:.4f}\n")
            self.compare_text.insert(tk.END, "\n")
    
    def compare_models(slef, image):
   
        results = {}
        for model_name in MODELS:
            try:
                top_preds = predict(image, model_name)[:1]
                results[model_name] = top_preds
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                results[model_name] = []
        
        return results


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
