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
# DESIGN SYSTEM — fonts, tokens, components
# =========================================================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0B1220;
            --surface: #121A2C;
            --surface-2: #182238;
            --border: rgba(148,163,184,0.14);
            --text: #E7ECF5;
            --text-muted: #8B95A7;
            --teal: #2DD4BF;
            --violet: #8B7CF6;
            --skin: #E8A87C;
            --green: #34D399;
            --yellow: #FBBF24;
            --red: #F87171;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }

        /* ---- Hero ---- */
        .hero-row { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
        .hero-mark {
            width: 68px; height: 68px; min-width: 68px;
            border-radius: 18px;
            background: linear-gradient(145deg, var(--teal) 0%, var(--violet) 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.9rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.06), 0 8px 24px -8px rgba(45,212,191,0.35);
        }
        .hero-row .hero-title,
        div[data-testid="stMarkdownContainer"] p.hero-title {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 3.4rem !important;
            color: var(--text) !important;
            letter-spacing: -0.03em !important;
            line-height: 1 !important;
            margin: 0 !important;
        }
        .hero-sub {
            color: var(--text-muted);
            font-size: 1.02rem;
            max-width: 640px;
            line-height: 1.55;
            margin: 10px 0 22px 0;
        }

        /* ---- Stat strip ---- */
        .stat-strip { display: flex; gap: 0; margin-bottom: 26px; flex-wrap: wrap; }
        .stat-item { padding: 4px 28px 4px 0; margin-right: 28px; border-right: 1px solid var(--border); }
        .stat-item:last-child { border-right: none; }
        .stat-num {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700; font-size: 1.6rem; color: var(--teal);
            line-height: 1;
        }
        .stat-label { color: var(--text-muted); font-size: 0.82rem; margin-top: 4px; }

        /* ---- Cards ---- */
        .panel {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px 24px;
        }
        .disclaimer {
            background-color: rgba(139,124,246,0.08);
            border: 1px solid rgba(139,124,246,0.28);
            border-left: 3px solid var(--violet);
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 0.92rem;
            color: var(--text);
            margin-bottom: 22px;
        }

        /* ---- Class mini-cards ---- */
        .class-card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--risk-color, var(--teal));
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .class-card h5 { margin: 0 0 4px 0; font-size: 0.98rem; color: var(--text); }
        .class-card .risk-tag { font-size: 0.76rem; font-weight: 600; margin-bottom: 6px; display: inline-block; }
        .class-card p { color: var(--text-muted); font-size: 0.85rem; margin: 4px 0 0 0; line-height: 1.45; }

        /* ---- Sidebar cards ---- */
        section[data-testid="stSidebar"] { background-color: var(--surface); }
        .side-card {
            background-color: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
        }
        .side-card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600; font-size: 0.95rem;
            color: var(--text); margin-bottom: 8px;
            display: flex; align-items: center; gap: 8px;
        }
        .side-card p { color: var(--text-muted); font-size: 0.86rem; line-height: 1.5; margin: 0; }

        .step-row { display: flex; gap: 12px; margin-bottom: 14px; align-items: flex-start; }
        .step-num {
            width: 22px; height: 22px; min-width: 22px; border-radius: 50%;
            background: rgba(45,212,191,0.15); color: var(--teal);
            font-size: 0.76rem; font-weight: 700;
            display: flex; align-items: center; justify-content: center;
            margin-top: 1px;
        }
        .step-text { color: var(--text-muted); font-size: 0.85rem; line-height: 1.45; }
        .step-text b { color: var(--text); }

        /* ---- Result header ---- */
        .result-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.55rem; color: var(--text); margin: 2px 0 8px 0; }
        .risk-pill {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 5px 14px; border-radius: 20px;
            font-size: 0.82rem; font-weight: 600;
        }

        .footer-note { text-align: center; color: var(--text-muted); font-size: 0.82rem; margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border); }

        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-family: 'Space Grotesk', sans-serif; }
        hr { border-color: var(--border) !important; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# CLASS REFERENCE DATA
# =========================================================
CLASS_DETAILS = {
    "nv": {
        "full_name": "Melanocytic Nevi (Common Mole)",
        "risk": "benign", "risk_label": "Benign", "color": "var(--green)",
        "description": "A common, ordinary mole made of pigment-producing cells. The vast majority of these are completely harmless and are the most frequently occurring skin lesion type.",
        "guidance": "Generally no action needed. Keep an eye on it for any changes in size, shape, or color over time (the ABCDE rule)."
    },
    "mel": {
        "full_name": "Melanoma",
        "risk": "danger", "risk_label": "Malignant — High Priority", "color": "var(--red)",
        "description": "The most dangerous form of skin cancer. It develops in melanocytes (pigment cells) and can spread to other parts of the body if not caught early.",
        "guidance": "See a dermatologist as soon as possible for a professional evaluation and possible biopsy. Early detection significantly improves outcomes."
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesion",
        "risk": "benign", "risk_label": "Benign", "color": "var(--green)",
        "description": "A group of non-cancerous skin growths, including seborrheic keratoses and solar lentigines, often appearing as waxy or scaly raised patches.",
        "guidance": "Usually harmless. A dermatologist visit is only needed if it becomes irritated, itchy, or changes rapidly."
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk": "danger", "risk_label": "Malignant — Needs Attention", "color": "var(--red)",
        "description": "The most common form of skin cancer. It grows slowly and rarely spreads to other parts of the body, but can cause local tissue damage if untreated.",
        "guidance": "Consult a dermatologist for evaluation and treatment. Highly treatable when caught early."
    },
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "risk": "caution", "risk_label": "Pre-Cancerous", "color": "var(--yellow)",
        "description": "Rough, scaly patches caused by long-term sun damage. These are considered pre-cancerous and can potentially develop into squamous cell carcinoma if left untreated.",
        "guidance": "A dermatology check-up is recommended. Treatable at this stage with simple procedures."
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "risk": "benign", "risk_label": "Benign", "color": "var(--green)",
        "description": "Skin marks caused by blood vessel abnormalities, such as angiomas or hemorrhages, typically appearing as red or purple spots.",
        "guidance": "Usually harmless and cosmetic in nature. Consult a doctor only if it changes suddenly or bleeds."
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk": "benign", "risk_label": "Benign", "color": "var(--green)",
        "description": "A common, firm, non-cancerous skin nodule, often caused by minor skin injuries like insect bites. Usually small and brownish.",
        "guidance": "No treatment necessary in most cases. Can be removed for cosmetic reasons if desired."
    },
}
RISK_ICON = {"benign": "🟢", "caution": "🟡", "danger": "🔴"}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
        <div class="hero-row" style="margin-bottom:18px;">
            <div class="hero-mark" style="width:40px;height:40px;min-width:40px;font-size:1.1rem;">🔬</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;color:var(--text);">LesionLens</div>
                <div style="color:var(--text-muted);font-size:0.78rem;">Skin Lesion Classifier</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="side-card">
            <div class="side-card-title">📌 About this project</div>
            <p>LesionLens analyzes dermoscopic images and classifies them into
            7 diagnostic categories, combining classical image preprocessing
            with a transfer-learning CNN and explainable AI.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="side-card">
            <div class="side-card-title">🧠 How it works</div>
            <div class="step-row">
                <div class="step-num">1</div>
                <div class="step-text"><b>Preprocess</b> — hair and artifacts are removed using classical computer vision.</div>
            </div>
            <div class="step-row">
                <div class="step-num">2</div>
                <div class="step-text"><b>Classify</b> — an EfficientNet-B0 model predicts the lesion type.</div>
            </div>
            <div class="step-row" style="margin-bottom:0;">
                <div class="step-num">3</div>
                <div class="step-text"><b>Explain</b> — Grad-CAM highlights the region the model focused on.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="side-card" style="margin-bottom:6px;">
            <div class="side-card-title">📊 Model performance</div>
        </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Test Accuracy", "78.3%")
    c2.metric("ROC-AUC", "0.956")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.caption("Dataset: HAM10000 (Kaggle) · Academic PRCV project")

# =========================================================
# HERO
# =========================================================
st.markdown("""
    <div class="hero-row">
        <div class="hero-mark">🔬</div>
        <p class="hero-title">LesionLens</p>
    </div>
    <p class="hero-sub">
        Upload a dermoscopic image and get an AI classification across 7 lesion types,
        a full confidence breakdown, and a visual explanation of exactly which region
        of the image drove the prediction.
    </p>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-num">7</div><div class="stat-label">Lesion classes detected</div></div>
        <div class="stat-item"><div class="stat-num">78.3%</div><div class="stat-label">Test set accuracy</div></div>
        <div class="stat-item"><div class="stat-num">10,015</div><div class="stat-label">Training images (HAM10000)</div></div>
        <div class="stat-item"><div class="stat-num">0.956</div><div class="stat-label">Macro ROC-AUC</div></div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Educational tool only — not a medical diagnosis.</b>
        Built for academic demonstration. Always consult a licensed dermatologist for any real skin concern.
    </div>
""", unsafe_allow_html=True)

# =========================================================
# THE 7 CLASSES — REFERENCE GUIDE
# =========================================================
with st.expander("📖 What are the 7 conditions this model detects?", expanded=False):
    cols = st.columns(3)
    for i, (code, info) in enumerate(CLASS_DETAILS.items()):
        with cols[i % 3]:
            st.markdown(f"""
                <div class="class-card" style="--risk-color:{info['color']};">
                    <h5>{info['full_name']}</h5>
                    <span class="risk-tag" style="color:{info['color']};">{RISK_ICON[info['risk']]} {info['risk_label']}</span>
                    <p>{info['description']}</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("📤 Upload a skin lesion image (JPG or PNG)", type=["jpg", "jpeg", "png"])

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

                    with col2:
                        st.markdown(f"""
                            <div class="panel">
                                <div style="color:var(--text-muted);font-size:0.85rem;">Prediction Result</div>
                                <div class="result-name">{pred_info['full_name']}</div>
                                <span class="risk-pill" style="background:{pred_info['color']}22;color:{pred_info['color']};border:1px solid {pred_info['color']}55;">
                                    {RISK_ICON[pred_info['risk']]} {pred_info['risk_label']}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("Confidence", f"{confidence_val}%")
                        m2.metric("Category", pred_info["risk_label"].split(" —")[0])
                        st.progress(confidence_val / 100)

                        if confidence_val < 60:
                            st.info("ℹ️ Low-confidence prediction — the model is uncertain. Treat this result with extra caution.")
                        elif confidence_val < 80:
                            st.info("ℹ️ Moderate-confidence prediction.")

                    st.markdown("### 💬 What Does This Mean?")
                    e1, e2 = st.columns(2)
                    with e1:
                        st.markdown("**About this condition**")
                        st.write(pred_info["description"])
                    with e2:
                        st.markdown("**Recommended next step**")
                        st.write(pred_info["guidance"])

                    st.divider()

                    st.markdown("### 📊 Full Probability Breakdown")
                    st.caption("How confident the model was across all 7 possible classes.")

                    probs_df = pd.DataFrame(result["all_probabilities"].items(), columns=["Class", "Probability (%)"])
                    probs_df["Full Name"] = probs_df["Class"].map(lambda c: CLASS_DETAILS[c]["full_name"])

                    fig = px.bar(
                        probs_df, x="Class", y="Probability (%)",
                        color="Probability (%)", color_continuous_scale=["#182238", "#2DD4BF"],
                        text="Probability (%)", hover_data={"Full Name": True, "Class": False}
                    )
                    fig.update_layout(
                        yaxis_range=[0, 100], showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=13, color="#E7ECF5")
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

                    st.markdown("### 🎯 Explainability — Where the Model Looked")
                    st.caption("Grad-CAM highlights the regions that most influenced the prediction. Red/yellow = high influence.")

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
                st.error("❌ Could not connect to the backend API. Make sure the FastAPI server is running at http://127.0.0.1:8000")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<p class="footer-note">LesionLens · Built with PyTorch, FastAPI & Streamlit · HAM10000 Dataset · Academic PRCV Project</p>',
    unsafe_allow_html=True
)