
   # 🧠 Amazon CNN Product Image Classifier with LIME Explainer

This professional-grade AI solution classifies product images (e.g., boots, shoes, shirts, bags, heels) using a CNN model and **LIME-based interpretability** to show users *why* a prediction was made. Ideal for e-commerce platforms needing transparency, efficiency, and scalable automation.

---

## 🔗 Live App & GitHub Repository

- 🎯 [**Streamlit App**](https://cnn-image-lime.streamlit.app)  
- 📁 [**GitHub Repository**](https://github.com/SweetySeelam2/Amazon-CNN-Product-Classification)

---

## 📂 Project Overview

| Feature                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| 🔍 **Model**               | CNN (Conv2D + MaxPooling + Dropout + Dense Layers)                         |
| 🧠 **Explainability**      | LIME (Local Interpretable Model-Agnostic Explanations)                     |
| 🧾 **Use Case**            | Explainable image classification for product catalogs                      |
| 🌐 **Interface**           | Streamlit app to upload images and view prediction + heatmap explanation   |

---

## 💼 Business Problem

Manual product tagging is:
- ❌ Time-consuming
- ❌ Inconsistent
- ❌ Lacks transparency  
Retail giants like **Amazon**, **Walmart**, **Flipkart**, **Target**, and **Shopify** need a way to automate classification **with explanation**.

---

## ✅ Solution: This Project

This app allows non-technical users (e.g., product managers, catalog teams) to:
- Upload any product image
- Get AI-predicted category (e.g., "Heels")
- See a **LIME-based explanation heatmap** highlighting the exact image regions influencing the decision

---

## 💰 Business Impact

| Metric                          | Value Estimate                      |
|----------------------------------|-------------------------------------|
| 💲 Manual Tagging Cost           | ~$1.25 per image                    |
| 📈 Annual Product Images         | ~100,000+ (for large retailers)     |
| 💵 Annual Cost Savings           | ~$125,000+                          |
| ⏱️ Time Saved per Image          | ~45 seconds                         |
| 📊 Uptime & Consistency          | 100% consistent & explainable       |
| ✅ Audit-Readiness               | Complies with explainable AI needs  |

---

## 🧪 Model Results & Interpretation

- ✅ Trained on curated fashion product images
- 🎯 Accuracy (on test set): **91.6%**
- 🔍 Confidence score (example prediction): **95.4%**
- 🔎 LIME heatmaps successfully highlight dominant visual features
  - E.g., shoe sole, heel, or neckline

---

## 🧠 Conclusion

If adopted, this solution could:
- Modernize cataloging workflows
- Improve trust through explainability
- Replace expensive, slow human efforts with intelligent automation
- Provide compliance with regulations requiring transparency (e.g., EU AI Act)

---

## 🧾 Business Recommendations

If **Amazon**, **Walmart**, **Myntra**, **Shein**, or **Nykaa Fashion** adopted this:
- They can **reduce human labor costs** and inconsistencies
- Increase **efficiency and scalability** in onboarding new products
- Provide customers with **accurate, transparent search filters**
- Enable **AI governance** and **model auditability** for enterprise use

---

## 🛠 Installation & Run Instructions

```bash
# Create virtual environment (optional)
python -m venv cnn_lime_env
source cnn_lime_env/bin/activate  # or cnn_lime_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit App
streamlit run app.py
```
---

## 👩‍💼 About the Author    

**Sweety Seelam** | Business Analyst | Aspiring Data Scientist | Passionate about building end-to-end ML solutions for real-world problems                                                                                                      
                                                                                                                                           
Email: sweetyseelam2@gmail.com                                                   

🔗 **Profile Links**                                                                                                                                                                       
[Portfolio Website](https://sweetyseelam2.github.io/SweetySeelam.github.io/)                                                         
[LinkedIn](https://www.linkedin.com/in/sweetyrao670/)                                                                   
[GitHub](https://github.com/SweetySeelam2)                                                             
[Medium](https://medium.com/@sweetyseelam)

---

## 🔐 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. All rights reserved.

This project, including its source code, trained models, datasets (where applicable), visuals, and dashboard assets, is protected under copyright and made available for educational and demonstrative purposes only.

Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
