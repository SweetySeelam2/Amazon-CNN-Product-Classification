import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === CONFIG ===
MODEL_PATH = "cnn_baseline_final_v2.h5"
TEST_CSV = "data/clean_test.csv"
IMAGE_DIR = "data/images"
LABEL_JSON = "index_to_label.json"
IMG_SIZE = (224, 224)

# === Load model and label index ===
model = load_model(MODEL_PATH)
with open(LABEL_JSON, "r") as f:
    index_to_label = json.load(f)

# === Streamlit UI ===
st.set_page_config(layout="centered", page_title="🧠 Product Classifier with LIME")
st.title("🧠 Amazon Product Image Classifier with LIME Explanation")

# === Select input ===
use_sample = st.toggle("Use a random test image from dataset")
if use_sample:
    test_df = pd.read_csv(TEST_CSV)
    row = test_df.sample(1, random_state=42).iloc[0]
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    true_label = row['label']
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_path = os.path.join("uploaded_img.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())
        true_label = "Unknown"
    else:
        st.stop()

# === Preprocess ===
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# === Predict ===
pred = model.predict(img_input)
predicted_class_idx = int(np.argmax(pred))
predicted_label = index_to_label[str(predicted_class_idx)]
confidence = round(100 * np.max(pred), 2)

# === Display prediction ===
st.image(img_path, caption="Input Image", width=300)
st.markdown(f"✅ **Predicted Label:** `{predicted_label}`")
st.markdown(f"📊 **Confidence:** `{confidence}%`")
st.markdown(f"🎯 **True Label:** `{true_label}`")

# === LIME Explanation ===
st.markdown("## 🔍 LIME Explanation")
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img_array.astype('double'),
    classifier_fn=lambda x: model.predict(x),
    top_labels=1,
    hide_color=0,
    num_samples=1000
)
temp, mask = explanation.get_image_and_mask(
    predicted_class_idx,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

fig, ax = plt.subplots()
ax.imshow(mark_boundaries(temp / 255.0, mask))
ax.axis("off")
st.pyplot(fig)

# Save & download
lime_filename = f"LIME_Explanation_{predicted_label}_img.png"
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
    This model, code, and app UI are protected under copyright law. <br>
    Any unauthorized reproduction or reuse is strictly prohibited.
    </p>
    """,
    unsafe_allow_html=True
)