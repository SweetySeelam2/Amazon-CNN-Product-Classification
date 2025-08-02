import streamlit as st
import numpy as np
import os
import requests
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import glob

# === CONFIG ===
MODEL_PATH = "cnn_baseline_final_v2.h5"
MODEL_RELEASE_URL = "https://huggingface.co/spaces/sweetyseelam/Amazon-CNN-Product-Classification/resolve/main/cnn_baseline_final_v2.h5"
SAMPLE_IMAGES_DIR = "sample_images"

# Github RAW links (replace with YOUR username/repo as needed)
LABEL_JSON = "index_to_label.json"
LABEL_JSON_URL = "https://raw.githubusercontent.com/SweetySeelam2/Amazon-CNN-Product-Classification/main/index_to_label.json"
CLEAN_TEST_CSV = "clean_test.csv"
CLEAN_TEST_CSV_URL = "https://raw.githubusercontent.com/SweetySeelam2/Amazon-CNN-Product-Classification/main/clean_test.csv"
REQUIREMENTS_TXT = "requirements.txt"
REQUIREMENTS_TXT_URL = "https://raw.githubusercontent.com/SweetySeelam2/Amazon-CNN-Product-Classification/main/requirements.txt"

IMG_SIZE = (224, 224)

# === Streamlit UI ===
st.set_page_config(page_title="🧠 Amazon Product Classifier", layout="centered")
st.title("🧠 Amazon Product Image Classifier with LIME")
st.markdown("Upload a product image **or** select a sample image from the project to get the predicted product category and visual explanation.")

# === Utility: Download from URL if missing ===
def download_if_missing(local_path, url, is_binary=False):
    if not os.path.exists(local_path):
        st.warning(f"🔄 {os.path.basename(local_path)} not found. Downloading from source...")
        with st.spinner(f"Downloading {os.path.basename(local_path)} ..."):
            r = requests.get(url, stream=is_binary)
            if r.status_code == 200:
                mode = "wb" if is_binary else "w"
                with open(local_path, mode) as f:
                    if is_binary:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    else:
                        f.write(r.text)
                st.success(f"✅ Downloaded {os.path.basename(local_path)}.")
            else:
                st.error(f"❌ Failed to download {os.path.basename(local_path)} from {url}")
                st.stop()

# === Download/check files ===
download_if_missing(MODEL_PATH, MODEL_RELEASE_URL, is_binary=True)
download_if_missing(LABEL_JSON, LABEL_JSON_URL)
download_if_missing(CLEAN_TEST_CSV, CLEAN_TEST_CSV_URL)
download_if_missing(REQUIREMENTS_TXT, REQUIREMENTS_TXT_URL)

# === Load model and label mapping ===
model = load_model(MODEL_PATH)
with open(LABEL_JSON, "r") as f:
    index_to_label = json.load(f)

# === Find all sample images in sample_images directory ===
sample_img_files = glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpg")) + \
                   glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.jpeg")) + \
                   glob.glob(os.path.join(SAMPLE_IMAGES_DIR, "*.png"))
sample_files = [os.path.basename(f) for f in sample_img_files]

# === User input: Upload or select a sample ===
uploaded_file = st.file_uploader("📤 Upload a product image", type=["jpg", "jpeg", "png"])

st.markdown("### OR")
selected_sample = st.selectbox("Or select a sample image from the project:", ["None"] + sample_files)

img_path = None
true_label = "Unknown"  # Default label

if uploaded_file:
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())
elif selected_sample != "None":
    img_path = os.path.join(SAMPLE_IMAGES_DIR, selected_sample)
else:
    st.info("Upload a product image **or** select a sample image above to begin.")
    st.stop()

# === Preprocess image ===
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# === Prediction ===
pred = model.predict(img_input)
pred_idx = int(np.argmax(pred))
pred_label = index_to_label[str(pred_idx)]
confidence = round(100 * np.max(pred), 2)

# === Display results ===
st.image(img_path, caption="Input Image", width=300)
st.markdown(f"✅ **Predicted Label:** `{pred_label}`")
st.markdown(f"📊 **Confidence:** `{confidence}%`")
st.markdown(f"🎯 **True Label:** `{true_label}`")

# === LIME Explanation ===
st.markdown("## 🔍 LIME Visual Explanation")
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img_array.astype('double'),
    classifier_fn=lambda x: model.predict(x),
    top_labels=1,
    hide_color=0,
    num_samples=1000
)
temp, mask = explanation.get_image_and_mask(
    pred_idx,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

fig, ax = plt.subplots()
ax.imshow(mark_boundaries(temp / 255.0, mask))
ax.axis("off")
st.pyplot(fig)

# === Download LIME output ===
lime_filename = f"LIME_Explanation_{pred_label}_img.png"
fig.savefig(lime_filename, bbox_inches='tight')
with open(lime_filename, "rb") as f:
    st.download_button("📥 Download LIME Heatmap", f, file_name=lime_filename)

# === Footer ===
st.markdown("---")
st.caption("App by Sweety Seelam")
st.markdown(
    """
    <p style='text-align: center; font-size: 12px;'>
    © 2025 <strong>Sweety Seelam</strong>. All rights reserved. <br>
    Model and code protected under copyright. Reuse without permission is prohibited.
    </p>
    """,
    unsafe_allow_html=True
)