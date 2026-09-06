# 🫁 PulmoScan — Pneumonia Detection with Grad-CAM Explainability

A deep learning system that detects Pneumonia from chest X-rays using a CNN, with **Grad-CAM visualization** to explain which regions of the X-ray influenced the model's decision.

## 🚀 Live Demo
https://pulmoscan-mmbwwqstchebvsbgzpkz8o.streamlit.app

## 📌 Overview
This project trains a Convolutional Neural Network (CNN) to classify chest X-rays as Normal or Pneumonia, and goes a step further than a typical classifier by adding **Grad-CAM (Gradient-weighted Class Activation Mapping)** — a technique that visualizes exactly which parts of the X-ray the model focused on to make its prediction.

## 🧠 Why Explainability Matters
Most beginner medical imaging projects stop at reporting accuracy. In a medical context, knowing *why* a model made a decision is as important as the decision itself. Grad-CAM was used specifically to verify that the model is learning genuine lung-pathology features rather than shortcut signals (e.g., bone edges, medical equipment in the image).

## 📊 Dataset
- Chest X-Ray Images (Pneumonia) — Kaggle
- Train: 5,216 images (1,341 Normal / 3,875 Pneumonia)
- Test: 624 images (234 Normal / 390 Pneumonia)
- Note: significant class imbalance in training data, handled using computed class weights

## 🛠️ Tech Stack
- Python, TensorFlow / Keras
- OpenCV (Grad-CAM heatmap generation and overlay)
- NumPy, Matplotlib, Seaborn
- Streamlit (deployment)

## 📈 Model Performance
| Metric | Normal | Pneumonia |
|---|---|---|
| Precision | 0.95 | 0.83 |
| Recall | 0.66 | 0.98 |
| F1-score | 0.78 | 0.90 |

Overall accuracy: 86%

**Why recall matters more here:** In medical screening, missing a real Pneumonia case (false negative) is far more dangerous than a false alarm. The model achieves 98% recall on Pneumonia, meaning it misses very few actual cases, at the cost of some false positives on Normal cases — a reasonable and deliberate trade-off for this kind of screening tool.

## ⚠️ Known Limitations
- Class imbalance in the training set (Pneumonia images outnumber Normal ~2.9:1) was addressed with class weighting, but Normal-class recall (66%) still has room for improvement.
- Grad-CAM visualizations sometimes show partial attention on bone structures and central chest area rather than purely lung tissue — a possible sign of the model using some shortcut features. This was identified through Grad-CAM analysis and is an area for further investigation (e.g., via transfer learning with a pretrained backbone).
- The validation split in the original dataset was too small (16 images) to be statistically meaningful; the test set was used for realistic evaluation instead.
- This is a student/educational project and is **not a certified diagnostic tool**.

## 🏃 How to Run Locally
```bash
git clone https://github.com/kdev76979-mac/PulmoScan.git
cd PulmoScan
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Download the dataset from Kaggle (Chest X-Ray Images - Pneumonia)
# and place it under data/raw/chest_xray/
# Run notebooks/eda.ipynb to train and generate models/pneumonia_model.keras
streamlit run app/app.py
```

## 📂 Project Structure