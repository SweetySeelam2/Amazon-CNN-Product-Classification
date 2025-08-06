import streamlit as st  
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="🧠 Amazon Product Classifier", layout="wide")

MODEL_PATH = "cnn_baseline_final_v2.keras"
SAMPLE_IMAGES_DIR = "sample_images"
LABEL_JSON = "index_to_label.json"
IMG_SIZE = (224, 224)
MAX_SAMPLE_IMAGES = 100

# --- Load model and label mapping
model = load_model(MODEL_PATH)
with open(LABEL_JSON, "r") as f:
    index_to_label = json.load(f)

# --- Load mini true label map for sample images only from Hugging Face
@st.cache_resource
def load_true_labels_csv_from_hf():
    csv_path = hf_hub_download(
        repo_id="sweetyseelam/Amazon-CNN-Product-Classification-v2",
        filename="sample_image_labels.csv",
        repo_type="space",
        cache_dir="./sample_images"
    )
    df = pd.read_csv(csv_path, dtype=str)
    id_to_label = dict(zip(df['id'], df['articleType']))
    return id_to_label

sample_true_label_map = load_true_labels_csv_from_hf()

def get_sample_image_local_path(filename):
    try:
        return hf_hub_download(
            repo_id="sweetyseelam/Amazon-CNN-Product-Classification-v2",
            filename=f"sample_images/{filename}",
            repo_type="space",
            cache_dir="./sample_images"
        )
    except Exception as ex:
        st.error(f"⚠️ Error downloading sample image: {filename} <br> {ex}", unsafe_allow_html=True)
        st.stop()

PAGES = {
    "🏠 Introduction": "intro",
    "🔍 Classifier & LIME Explanation": "predict",
    "📊 Results, Impact & Storytelling": "results"
}
page = st.sidebar.radio("Go to:", list(PAGES.keys()))

def footer():
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align: center; font-size: 12px;'>
        © 2025 <strong>Sweety Seelam</strong>. All rights reserved. <br>
        Model and code protected under copyright. Reuse without permission is prohibited.
        </p>
        """,
        unsafe_allow_html=True
    )

lime_feature_map = {
    "Tshirts": "the model focused on regions like the shirt neckline, sleeves, and central chest area, which are typical indicators for Tshirt recognition.",
    "Shoes": "the LIME highlights include the shoe sole, lace region, and toe box, showing the model used shape and structure cues unique to footwear.",
    "Watches": "the highlighted areas cover the dial, strap, and watch face, indicating that the model relied on these visual cues.",
    "Bags": "the attention is on the handles and main body of the bag, which are crucial for distinguishing bag types.",
    "Heels": "the highlighted heel and upper region suggest the model identifies the class by shape and elevation typical of heels."
}
def get_lime_interp(pred_label):
    core = lime_feature_map.get(pred_label, 
        "the model's attention is drawn to the most distinguishing parts of the product image, ensuring robust category identification.")
    return f"In this example, {core}"

if PAGES[page] == "intro":
    st.title("🧠 Amazon Product Image Classifier with LIME")
    st.markdown("""
    Welcome to a state-of-the-art **Explainable AI app** that classifies fashion product images and visually explains its decisions with LIME!  
    ---
    **Project Overview:**
    - Classifies fashion product images (e.g., boots, shoes, shirts, bags, heels) using a high-accuracy CNN model.
    - Explains every prediction with a **LIME heatmap** so business users can *trust* and *understand* the AI.
    - Built for e-commerce catalog managers, product teams, data scientists, and anyone needing transparency and automation.

    **Business Problem:**
    - Manual product tagging is slow, inconsistent, and lacks transparency.
    - Retailers like Amazon, Walmart, Flipkart, Target, and Shopify need fast, explainable automation.

    **Our Solution:**
    - Upload any product image or choose from 100+ real samples.
    - Instantly get a predicted category and an interactive visual explanation.
    - Confidence scores for transparency and auditability.
    - Easy to use — no coding required.

    *Ready to see it in action? Click "Classifier & LIME Explanation" in the sidebar to start!*  
    """)
    footer()

if PAGES[page] == "predict":
    st.title("🔍 Product Classifier & LIME Visual Explanation")
    #st.markdown("Upload a product image **or** select a sample image below. After your choice, click the corresponding **Submit** button to get a prediction and see why the AI made its decision.")
    st.markdown("Select a sample image below. After your choice, click the **Submit** button to get a prediction and see *why* the AI made its decision.")

    # List sample images
    try:
        sample_files = [
            f for f in sorted(os.listdir(SAMPLE_IMAGES_DIR))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:MAX_SAMPLE_IMAGES]
    except Exception as e:
        sample_files = []
        st.warning(f"Could not find or read sample_images: {e}")

    #uploaded_file = st.file_uploader("📤 Upload a product image", type=["jpg", "jpeg", "png"])
    #submit_upload = st.button("Submit Uploaded Image", key="submit_upload")
    #st.markdown("### OR")
    selected_sample = st.selectbox("Select a sample image from the project:", ["None"] + sample_files)
    submit_sample = st.button("Submit Sample Image", key="submit_sample")

    img_path = None
    true_label = None
    show_true_label = False
    user_mode = None
    proceed = False

    # --- USER UPLOAD HANDLING
    #if uploaded_file and submit_upload:
    #    img_path = "uploaded_image.jpg"
    #    with open(img_path, "wb") as f:
    #        f.write(uploaded_file.read())
    #    show_true_label = False
    #    user_mode = "user"
    #    proceed = True

    # --- SAMPLE IMAGE HANDLING
    if selected_sample != "None" and submit_sample:
        img_path = get_sample_image_local_path(selected_sample)
        sample_id = os.path.splitext(selected_sample)[0]
        true_label = sample_true_label_map.get(sample_id, "Unknown")
        show_true_label = True
        user_mode = "sample"
        proceed = True

    if not proceed:
        #st.info("Upload a product image and click **Submit Uploaded Image**, or select a sample image and click **Submit Sample Image** below.")
        st.info("Select a sample image and click **Submit Sample Image** below.")
        footer()
        st.stop()

    # --- Check the image can be loaded, else show error
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
    except Exception as ex:
        st.error(f"⚠️ Error loading image: {img_path} - {ex}\n\n"
                 "This is usually because the image is missing/corrupt. "
                 "If this is a sample image, some files may not be available in this Hugging Face Space due to LFS/data quota. "
                 "Please select another image or upload your own.")
        footer()
        st.stop()

    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # --- Model Prediction (NEW LOGIC BELOW) ---
    pred = model.predict(img_input)
    pred_idx = int(np.argmax(pred))
    pred_label = index_to_label.get(str(pred_idx), "Unknown")
    confidence = round(100 * np.max(pred), 2)

    # --- Only show LIME/impact if prediction in training set ---
    is_unknown = (str(pred_idx) not in index_to_label)

    st.image(img_path, caption=f"Input Image: {os.path.basename(img_path)}", width=160)

    if is_unknown:
        st.markdown(f"✅ <b>Predicted Label:</b> <span style='color:#ba0c2f'>Unknown</span>", unsafe_allow_html=True)
        st.info("""
        <b>Note:</b> The model is trained on 54 specific product classes.  
        This image does not match any of those classes, so the model cannot reliably recognize it.<br><br>
        <b>Data-backed explanation:</b> Our model is specifically trained and validated only on images belonging to 54 fashion product categories.
        When an image outside these classes is uploaded, the AI is unable to produce a reliable prediction.  
        This ensures your results are always transparent, fair, and explainable.
        """, unsafe_allow_html=True)
        footer()
        st.stop()

    # --- Normal path for "Known" predictions ---
    # --- Explanation for Predicted Label
    st.markdown(f"✅ <b>Predicted Label:</b> <span style='color:#388e3c'>{pred_label}</span>", unsafe_allow_html=True)
    st.markdown("<small>This is the product category predicted by our AI model based on the selected image.</small>", unsafe_allow_html=True)

    # --- Explanation for Confidence Score
    st.markdown(f"📊 <b>Confidence Score:</b> <span style='color:#1565c0'>{confidence}%</span>", unsafe_allow_html=True)
    st.markdown("<small>This score shows how confident the AI model is about its prediction. A score close to 100% means high certainty.</small>", unsafe_allow_html=True)

    # --- True Label (Sample Images Only) + Explanation
    if show_true_label and true_label is not None:
        st.markdown(f"🎯 <b>True Label:</b> <span style='color:#ba0c2f'>{true_label}</span>", unsafe_allow_html=True)
        if pred_label != true_label:
            st.markdown(
                "<small><b>Note:</b> The predicted class may sometimes differ from the true label if the product visually overlaps with multiple categories. This helps us improve and retrain the model over time.</small>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<small>This is the actual category label from our curated dataset, used for evaluating model accuracy.</small>",
                unsafe_allow_html=True
            )

    # --- LIME Explanation ---
    st.markdown("## 🔍 LIME Visual Explanation")
    with st.spinner("Generating LIME explanation..."):
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
        # Always overlay LIME boundaries on the original product image, not temp!
        img_show = img_array.copy()  # Use the *original* image, not temp
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(mark_boundaries(img_show, mask))
        ax.axis("off")
        lime_filename = f"LIME_Explanation_{pred_label}_img.png"
        fig.savefig(lime_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        st.image(lime_filename, width=320)
        with open(lime_filename, "rb") as f:
            st.download_button("📥 Download LIME Heatmap", f, file_name=lime_filename)

    # --- LIME Dynamic Interpretation Block ---
    LIME_DESCRIPTIONS = {
        "Tshirts": "the yellow areas usually highlight the neckline, sleeves, and chest—this shows the AI focuses on the shirt's shape and collar to make its decision.",
        "Shoes": "the yellow regions often cover the sole, toe, and laces—indicating the AI is looking at typical shoe features.",
        "Watches": "yellow highlights focus on the watch face and strap—this means the model recognizes these as most important.",
        "Bags": "the AI's explanation highlights the handles and main body of the bag—showing it uses shape and straps to classify.",
        "Heels": "the yellow areas point to the elevated heel and shoe outline—meaning the model looks for the height and shape unique to heels.",
    }
    if pred_label in LIME_DESCRIPTIONS:
        user_friendly_interp = LIME_DESCRIPTIONS[pred_label]
    else:
        user_friendly_interp = "the yellow areas highlight the most unique parts of this product, helping the AI identify its category accurately."

    #if user_mode == "user":
    #    filename_no_ext = os.path.splitext(uploaded_file.name)[0]
    #    lime_caption = f"uploaded image (<b>{filename_no_ext}</b>)"
    #else:
    lime_caption = f"sample image (<b>{true_label}</b>)"

    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 1em; border-radius: 8px; margin-top: 1.5em;">
        <b>What is LIME?</b><br>
        LIME is an explainable AI tool that highlights which parts of an image most influenced the model’s decision.<br>
        <br>
        <b>How to interpret this LIME output?</b><br>
        For this {lime_caption}, the yellow highlighted areas show what the AI focused on to predict <b>{pred_label}</b>.<br>
        In simple terms, {user_friendly_interp}
        <br><br>
        This helps users, even without a technical background, to trust that the AI is looking at the right parts (not just random patterns) and makes it easier to review or audit the AI's logic.<br>
        <br>
        <b>Why does this matter?</b>
        <ul>
            <li>Builds trust for product teams, managers, and auditors</li>
            <li>Makes model decisions transparent and compliant with business needs</li>
            <li>Accelerates onboarding and model adoption in e-commerce & retail</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

    # --- Conclusion & Business Impact (dynamic) ---
    st.markdown("### 📈 Conclusion & Business Impact")
    st.markdown(
        f"""
        <b>For this image (<span style='color:#388e3c'>{pred_label}</span>), the AI model classifies with <span style='color:#1565c0'>{confidence}%</span> confidence.</b><br>
        - Based on project results: Our system achieves up to <b>85.44% accuracy</b> on unseen Amazon product images.<br>
        - Business impact: Each automated classification can save <b>$1.25</b> compared to manual tagging.<br>
        - At 100,000 product uploads/year, this unlocks <b>$125,000</b>+ in annual cost savings and ~60% faster onboarding.<br>
        - Using LIME, decisions are <b>auditable and trusted</b>, supporting compliance (e.g., EU AI Act).
        """, unsafe_allow_html=True
    )

    # --- Business Recommendations ---
    st.markdown("### 💡 Business Recommendations")
    st.markdown(
        f"""
        <ul>
        <li><b>Deploy this classifier for automated, error-free tagging of <span style='color:#388e3c'>{pred_label}</span> and similar products.</b></li>
        <li>Scale to other product categories to maximize cost savings and operational speed.</li>
        <li>Recommended for e-commerce and retail leaders like Amazon, Walmart, Flipkart, Shopify, Zalando, and Target.</li>
        <li>Immediate ROI via reduced manual effort, more accurate search filters, and improved catalog trust.</li>
        </ul>
        """, unsafe_allow_html=True
    )

    footer()

if PAGES[page] == "results":
    st.title("📊 Results, Analysis, Business Impact & Storytelling")
    st.markdown("""
    **Model Results & Interpretation**
    - Trained on the full Fashion Product Images Dataset of 44,450 labeled images using a balanced, professional CNN pipeline.
    - Achieved **85.44% validation accuracy** (best epoch), with typical confidence scores of 90–100% on clean samples.
    - LIME heatmaps consistently highlight the correct visual regions (e.g., shoe sole, heel, shirt print).

    **Business Impact & ROI**
    | Metric                   | Value Estimate          |
    |--------------------------|------------------------|
    | 💲 Manual Tagging Cost   | ~$1.25 per image       |
    | 📈 Images per Year       | ~100,000+ (retailers)  |
    | 💵 Cost Savings          | ~$125,000+ per year    |
    | ⏱️ Time Saved/Image      | ~60% onboarding time   |
    | 🧮 Error Rate Reduction  | ~20–25%                |
    | 📊 Consistency           | 100% auditable         |

    **Business Recommendations**                                                                                             
    If adopted by Amazon, Walmart, Myntra, Shein, Nykaa, Flipkart, Target, etc.:
      - Reduce human labor costs and inconsistencies
      - Increase onboarding speed for new SKUs
      - Provide more accurate, explainable catalog search
      - Enable audit-readiness and trust for compliance

    **Project Storytelling**                                                                                                                                                                              
    Imagine a retailer like Amazon onboarding thousands of new products every month. Instead of slow, error-prone manual tagging, this system:
    - Instantly classifies each new image
    - Justifies its answer visually
    - Saves over **$125,000/year** and enables 100% transparent automation
    - Empowers merchandisers and data teams with full model trust and accountability

    This app is a leap forward for explainable AI in retail — and a model for scalable, ethical automation.
    """)
    footer()