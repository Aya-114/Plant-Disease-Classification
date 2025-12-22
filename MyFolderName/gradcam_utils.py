import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create model to map last conv layer to predictions
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Get gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply pooled grads by conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap


def save_gradcam(image_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Heatmap to RGB
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    output = heatmap * alpha + img

    # Save
    cv2.imwrite(output_path, output)