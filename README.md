<div align="center">

# 🔬 LesionLens

### AI-Powered Skin Lesion Classification with Explainable Predictions

**Deep Learning · Computer Vision · Explainable AI · Full-Stack Deployment**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

*An end-to-end deep learning system that classifies dermoscopic skin lesion images into 7 diagnostic categories, explains its own predictions with Grad-CAM, and serves it all through a full-stack web application.*

</div>

---

## 📖 Overview

**LesionLens** is a complete computer vision pipeline built for classifying skin lesions using the **HAM10000** dataset — one of the most widely used benchmark datasets in dermatological AI research. The project goes beyond a simple "upload image → get label" classifier: it combines **classical image processing**, **transfer learning**, and **explainable AI**, wrapped in a polished full-stack web application.

This project was built as an academic **Pattern Recognition & Computer Vision (PRCV)** capstone, demonstrating the complete lifecycle of an applied ML system — from raw data to a deployable, explainable product.

> ⚠️ **Disclaimer:** LesionLens is an educational/academic tool and is **not** a medical diagnostic device. It should never be used as a substitute for professional medical advice.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧬 **7-Class Classification** | Classifies lesions into `melanoma`, `melanocytic nevi`, `basal cell carcinoma`, `actinic keratoses`, `benign keratosis`, `dermatofibroma`, and `vascular lesions` |
| 🧼 **Classical CV Preprocessing** | Custom DullRazor-style hair/artifact removal using morphological filtering (black-hat transform + inpainting) — before any deep learning touches the image |
| ⚖️ **Imbalance-Aware Training** | Class-weighted loss function to counter HAM10000's severe 58:1 class imbalance |
| 🧠 **Transfer Learning** | Fine-tuned EfficientNet-B0 (ImageNet-pretrained) for efficient, accurate classification |
| 🔥 **Grad-CAM Explainability** | Visualizes exactly which pixels influenced each prediction — verifying the model looks at lesions, not artifacts |
| ⚡ **FastAPI Backend** | REST API serving predictions and live-generated Grad-CAM heatmaps |
| 🎨 **Polished Streamlit UI** | Clean, designed frontend with confidence visualization, risk categorization, and plain-language explanations |
| 📊 **Rigorous Evaluation** | Full precision/recall/F1 per class, macro ROC-AUC, and confusion matrix analysis — not just accuracy |

---

## 🏗️ Architecture

```
┌─────────────────┐      HTTP POST       ┌──────────────────┐      Inference      ┌────────────────────┐
│                  │  (image upload)      │                  │   (forward pass)    │                    │
│  Streamlit UI    │ ───────────────────► │   FastAPI         │ ───────────────────► │  EfficientNet-B0   │
│  (Frontend)      │                      │   Backend         │                      │  (PyTorch Model)    │
│                  │ ◄─────────────────── │                  │ ◄─────────────────── │                    │
└─────────────────┘   JSON + Heatmap      └──────────────────┘    Prediction +       └────────────────────┘
                                                                    Grad-CAM
```

**Pipeline flow:** Image Upload → Hair/Artifact Removal (OpenCV) → Resize & Normalize → CNN Inference → Softmax Probabilities → Grad-CAM Heatmap Generation → Rendered Result

---

## 🧠 Model & Dataset

| Detail | Value |
|---|---|
| **Dataset** | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermoscopic images |
| **Architecture** | EfficientNet-B0 (transfer learning, ImageNet-pretrained) |
| **Input Size** | 224 × 224 RGB |
| **Split Strategy** | Lesion-ID-based stratified split (70/15/15) — prevents data leakage from repeated lesion photographs |
| **Loss Function** | Class-weighted Cross-Entropy Loss |
| **Optimizer** | Adam (lr=1e-4) with `ReduceLROnPlateau` scheduling |
| **Training Epochs** | 15 |

### 📊 Test Set Performance

| Metric | Score |
|---|---|
| **Accuracy** | 78.3% |
| **Macro F1-Score** | 0.689 |
| **Weighted F1-Score** | 0.794 |
| **Macro ROC-AUC** | **0.956** |

<details>
<summary><b>Full per-class classification report</b></summary>

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| akiec | 0.506 | 0.769 | 0.611 | 52 |
| bcc | 0.713 | 0.803 | 0.755 | 71 |
| bkl | 0.596 | 0.575 | 0.585 | 167 |
| df | 0.484 | 0.750 | 0.588 | 20 |
| **mel** | 0.459 | **0.641** | 0.535 | 167 |
| nv | 0.941 | 0.839 | 0.887 | 1004 |
| vasc | 0.826 | 0.905 | 0.864 | 21 |

</details>

**Key finding:** Grad-CAM analysis confirms the model consistently attends to the lesion region itself — even on misclassified examples — indicating that classification errors stem from genuine visual ambiguity between classes (e.g., melanoma vs. benign nevi), not from distraction by artifacts like hair or rulers.

---

## 📁 Project Structure

```
lesionlens/
├── backend/
│   ├── main.py              # FastAPI app — /predict and /gradcam endpoints
│   ├── model_utils.py        # Model loading, preprocessing, hair removal, Grad-CAM
│   └── requirements.txt      # Backend dependencies
│
├── frontend/
│   └── app.py                # Streamlit application (UI)
│
├── model/
│   ├── lesionlens_model.pth  # Trained EfficientNet-B0 weights
│   └── class_to_idx.json     # Class label mapping
│
├── notebook/
│   └── LesionLens_Notebook_Documentation.md   # Full training notebook (EDA → Evaluation)
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lesionlens.git
cd lesionlens
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
pip install streamlit requests plotly
```

### 4. Run the backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```
Backend runs at `http://127.0.0.1:8000` — interactive API docs available at `/docs`.

### 5. Run the frontend (Streamlit)

Open a **new terminal**, activate the virtual environment again, then:

```bash
cd frontend
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

> ⚠️ Both the backend **and** frontend must be running simultaneously for the app to work.

---

## 🧪 Tech Stack

**Machine Learning:** PyTorch · Torchvision · EfficientNet-B0 · Grad-CAM
**Computer Vision:** OpenCV (morphological hair removal, inpainting)
**Backend:** FastAPI · Uvicorn
**Frontend:** Streamlit · Plotly
**Data Handling:** Pandas · NumPy · Scikit-learn

---

## 🔭 Future Work

- [ ] **RAG-powered chatbot** — retrieval-augmented conversational layer allowing users to ask follow-up questions about their specific prediction, grounded in curated dermatological reference material and the model's own output
- [ ] Model ensembling (EfficientNet + ViT) for improved minority-class recall
- [ ] Deployment to a public cloud endpoint

---

## 📚 Dataset Citation

Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018).

---

## ⚖️ License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**Built as an academic PRCV project** · Powered by PyTorch, FastAPI & Streamlit

</div>
