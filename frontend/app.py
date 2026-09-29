import streamlit as st
import requests
from PIL import Image
import pandas as pd
import plotly.express as px
import io

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="LesionLens — Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_PREDICT_URL = "http://127.0.0.1:8000/predict"
API_GRADCAM_URL = "http://127.0.0.1:8000/gradcam"

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 50%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #9CA3AF;
            margin-top: 0px;
            margin-bottom: 1.2rem;
        }
        .info-card {
            background-color: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
        }
        .result-card {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 22px 26px;
            margin-top: 10px;
        }
        .class-chip {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 3px;
        }
        .chip-benign { background-color: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(34,197,94,0.4); }
        .chip-caution { background-color: rgba(250,204,21,0.15); color: #FACC15; border: 1px solid rgba(250,204,21,0.4); }
        .chip-danger { background-color: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(239,68,68,0.4); }
        .footer-note {
            text-align: center;
            color: #6B7280;
            font-size: 0.85rem;
            margin-top: 40px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.7rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# CLASS REFERENCE DATA
# (full names, plain-language descriptions, risk level, guidance)
# =========================================================
CLASS_DETAILS = {
    "nv": {
        "full_name": "Melanocytic Nevi (Common Mole)",
        "risk": "benign",
        "risk_label": "Benign",
        "description": "A common, ordinary mole made of pigment-producing cells. The vast majority of these are completely harmless and are the most frequently occurring skin lesion type.",
        "guidance": "Generally no action needed. Keep an eye on it for any changes in size, shape, or color over time (the ABCDE rule)."
    },
    "mel": {
        "full_name": "Melanoma",
        "risk": "danger",
        "risk_label": "Malignant — High Priority",
        "description": "The most dangerous form of skin cancer. It develops in melanocytes (pigment cells) and can spread to other parts of the body if not caught early.",
        "guidance": "See a dermatologist as soon as possible for a professional evaluation and possible biopsy. Early detection significantly improves outcomes."
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesion",
        "risk": "benign",
        "risk_label": "Benign",
        "description": "A group of non-cancerous skin growths, including seborrheic keratoses and solar lentigines, often appearing as waxy or scaly raised patches.",
        "guidance": "Usually harmless. A dermatologist visit is only needed if it becomes irritated, itchy, or changes rapidly."
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk": "danger",
        "risk_label": "Malignant — Needs Attention",
        "description": "The most common form of skin cancer. It grows slowly and rarely spreads to other parts of the body, but can cause local tissue damage if untreated.",
        "guidance": "Consult a dermatologist for evaluation and treatment. Highly treatable when caught early."
    },
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "risk": "caution",
        "risk_label": "Pre-Cancerous",
        "description": "Rough, scaly patches caused by long-term sun damage. These are considered pre-cancerous and can potentially develop into squamous cell carcinoma if left untreated.",
        "guidance": "A dermatology check-up is recommended. Treatable at this stage with simple procedures."
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "risk": "benign",
        "risk_label": "Benign",
        "description": "Skin marks caused by blood vessel abnormalities, such as angiomas or hemorrhages, typically appearing as red or purple spots.",
        "guidance": "Usually harmless and cosmetic in nature. Consult a doctor only if it changes suddenly or bleeds."
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk": "benign",
        "risk_label": "Benign",
        "description": "A common, firm, non-cancerous skin nodule, often caused by minor skin injuries like insect bites. Usually small and brownish.",
        "guidance": "No treatment necessary in most cases. Can be removed for cosmetic reasons if desired."
    },
}

RISK_CHIP_CLASS = {"benign": "chip-benign", "caution": "chip-caution", "danger": "chip-danger"}
RISK_ICON = {"benign": "🟢", "caution": "🟡", "danger": "🔴"}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🔬 LesionLens")
    st.caption("AI-Powered Skin Lesion Classifier")
    st.markdown("---")

    st.markdown("#### 📌 About This Project")
    st.markdown(
        """
        LesionLens uses a deep learning model to analyze dermoscopic
        images of skin lesions and classify them into **7 diagnostic
        categories**, combining classical computer vision preprocessing
        with a transfer-learning CNN and explainable AI.
        """
    )

    st.markdown("#### 🧠 How It Works")
    st.markdown(
        """
        1. **Preprocessing** — hair/artifact removal using classical CV
        2. **Classification** — EfficientNet-B0 (transfer learning)
        3. **Explainability** — Grad-CAM shows *where* the model looked
        """
    )

    st.markdown("#### 📊 Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("Test Accuracy", "78.3%")
    c2.metric("Macro ROC-AUC", "0.956")

    st.markdown("---")
    st.caption("Built as an academic PRCV project — Dataset: HAM10000 (Kaggle)")

# =========================================================
# HEADER
# =========================================================
st.markdown('<p class="main-header">🔬 LesionLens</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a skin lesion image to get an AI-powered classification, '
    'confidence breakdown, and a visual explanation of the model\'s reasoning.</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
    ⚠️ <b>Educational tool only — not a medical diagnosis.</b>
    This project is built for academic demonstration purposes. Always consult a
    licensed dermatologist for any real skin concern.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# THE 7 CLASSES — REFERENCE GUIDE
# =========================================================
with st.expander("📖 What are the 7 conditions this model detects?", expanded=False):
    st.markdown("This model classifies uploaded images into one of the following categories:")
    cols = st.columns(3)
    for i, (code, info) in enumerate(CLASS_DETAILS.items()):
        with cols[i % 3]:
            chip_class = RISK_CHIP_CLASS[info["risk"]]
            st.markdown(f"**{info['full_name']}**")
            st.markdown(
                f'<span class="class-chip {chip_class}">{RISK_ICON[info["risk"]]} {info["risk_label"]}</span>',
                unsafe_allow_html=True
            )
            st.caption(info["description"])
            st.markdown("")

st.markdown("---")

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "📤 Upload a skin lesion image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        analyze_clicked = st.button("🔍 Analyze Lesion", type="primary", use_container_width=True)

    if analyze_clicked:
        with st.spinner("Running preprocessing and classification..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(API_PREDICT_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    pred_code = result["predicted_class"]
                    pred_info = CLASS_DETAILS[pred_code]
                    confidence_val = result["confidence"]

                    # ---- Prediction Result Card ----
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### Prediction Result")

                        chip_class = RISK_CHIP_CLASS[pred_info["risk"]]
                        st.markdown(f"### {pred_info['full_name']}")
                        st.markdown(
                            f'<span class="class-chip {chip_class}">{RISK_ICON[pred_info["risk"]]} {pred_info["risk_label"]}</span>',
                            unsafe_allow_html=True
                        )

                        st.markdown("")
                        m1, m2 = st.columns(2)
                        m1.metric("Confidence", f"{confidence_val}%")
                        m2.metric("Risk Category", pred_info["risk_label"])

                        st.progress(confidence_val / 100)

                        if confidence_val < 60:
                            st.info("ℹ️ This is a **low-confidence** prediction — the model is uncertain. Treat this result with extra caution.")
                        elif confidence_val < 80:
                            st.info("ℹ️ This is a **moderate-confidence** prediction.")

                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---- Explanation Block ----
                    st.markdown("### 💬 What Does This Mean?")
                    e1, e2 = st.columns(2)
                    with e1:
                        st.markdown("**About this condition:**")
                        st.write(pred_info["description"])
                    with e2:
                        st.markdown("**Recommended next step:**")
                        st.write(pred_info["guidance"])

                    st.divider()

                    # ---- Probability Breakdown ----
                    st.markdown("### 📊 Full Probability Breakdown")
                    st.caption("How confident the model was across all 7 possible classes.")

                    probs_df = pd.DataFrame(
                        result["all_probabilities"].items(),
                        columns=["Class", "Probability (%)"]
                    )
                    probs_df["Full Name"] = probs_df["Class"].map(lambda c: CLASS_DETAILS[c]["full_name"])

                    fig = px.bar(
                        probs_df,
                        x="Class",
                        y="Probability (%)",
                        color="Probability (%)",
                        color_continuous_scale="Blues",
                        text="Probability (%)",
                        hover_data={"Full Name": True, "Class": False}
                    )
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=13)
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

                    # ---- Grad-CAM ----
                    st.markdown("### 🎯 Explainability — Where the Model Looked")
                    st.caption(
                        "Grad-CAM highlights the image regions that most influenced the prediction. "
                        "Red/yellow = high influence, blue/purple = low influence."
                    )

                    with st.spinner("Generating explainability heatmap..."):
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        gradcam_response = requests.post(API_GRADCAM_URL, files=files)

                        if gradcam_response.status_code == 200:
                            gradcam_image = Image.open(io.BytesIO(gradcam_response.content))
                            g1, g2 = st.columns(2)
                            with g1:
                                st.image(image, caption="Original", use_container_width=True)
                            with g2:
                                st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                        else:
                            st.warning("Could not generate Grad-CAM heatmap.")

                else:
                    st.error(f"API Error: {response.status_code} — {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to the backend API. "
                    "Make sure the FastAPI server is running at http://127.0.0.1:8000"
                )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<p class="footer-note">LesionLens · Built with PyTorch, FastAPI & Streamlit · '
    'HAM10000 Dataset · Academic PRCV Project</p>',
    unsafe_allow_html=True
)