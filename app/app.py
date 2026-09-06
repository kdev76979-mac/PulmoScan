import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 150

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/pneumonia_model.keras')
    return model

model = load_model()

def get_gradcam_heatmap(model, img_array, last_conv_layer_name='last_conv_layer'):
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    last_conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[last_conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    img = np.uint8(255 * img_array[0])
    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed

# ---------- UI ----------
st.title("🫁 PulmoScan — Pneumonia Detector")
st.write("Upload a chest X-ray image to detect Pneumonia, with Grad-CAM explainability.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # grayscale
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # channel dim
    img_array = np.expand_dims(img_array, axis=0)   # batch dim

    pred = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred

    heatmap = get_gradcam_heatmap(model, img_array)
    overlayed_img = overlay_gradcam(img_array, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.image(overlayed_img, caption="Grad-CAM Heatmap", use_container_width=True)

    if label == "PNEUMONIA":
        st.error(f"Prediction: **{label}** (Confidence: {confidence:.2%})")
    else:
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2%})")

    # st.caption("⚠️ This is a student project for educational purposes only, not a medical diagnostic tool.")