import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 150

st.set_page_config(page_title="PulmoScan", page_icon="🫁", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
.main-header {
    background-color: #0D6E68;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.main-header h1 { color: white; margin: 0; font-size: 28px; }
.main-header p { color: #D6EDE9; margin: 6px 0 0 0; font-size: 14px; }
.result-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 16px;
    border: 1px solid #E0E6E7;
}
.result-danger { background-color: #FDECEC; border-color: #E6C1C1; }
.result-safe { background-color: #E7F5F1; border-color: #B9DED4; }
.disclaimer { font-size: 12px; color: #6B7B7B; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🫁 PulmoScan</h1>
    <p>AI-assisted chest X-ray screening with explainable predictions</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/pneumonia_model.keras')

model = load_model()

def is_likely_xray(pil_img, threshold=15):
    """X-rays are near-grayscale (R≈G≈B). Regular photos are not."""
    rgb_img = pil_img.convert('RGB')
    img_array = np.array(rgb_img)
    r, g, b = img_array[:,:,0].astype(int), img_array[:,:,1].astype(int), img_array[:,:,2].astype(int)
    diff = (np.abs(r-g) + np.abs(g-b) + np.abs(r-b)) / 3
    return diff.mean() < threshold

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
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

# ---------- UI ----------
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file)

    if not is_likely_xray(original_img):
        st.markdown("""
        <div class="result-card result-danger">
            <b>⚠️ Invalid image</b><br>
            This doesn't look like a chest X-ray. PulmoScan only analyzes grayscale
            radiographs — please upload a valid chest X-ray image.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    img = original_img.convert('L')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred

    heatmap = get_gradcam_heatmap(model, img_array)
    overlayed_img = overlay_gradcam(img_array, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.image(overlayed_img, caption="Grad-CAM heatmap", use_container_width=True)

    css_class = "result-danger" if label == "PNEUMONIA" else "result-safe"
    icon = "⚠️" if label == "PNEUMONIA" else "✅"
    st.markdown(f"""
    <div class="result-card {css_class}">
        <b>{icon} Prediction: {label}</b><br>
        Confidence: {confidence:.2%}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="disclaimer">⚠️ Educational project — not a certified diagnostic tool.</p>', unsafe_allow_html=True)