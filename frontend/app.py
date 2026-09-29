# ============================================================
# LesionLens - Streamlit Frontend
# ============================================================
# This file creates the user interface for LesionLens.
#
# Main responsibilities:
# 1. Allow the user to upload a skin-lesion image
# 2. Send the image to the FastAPI backend
# 3. Display the model prediction
# 4. Display confidence and class probabilities
# 5. Warn the user when the prediction is borderline
# ============================================================


# ============================================================
# Imports
# ============================================================

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from PIL import Image


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="LesionLens — Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)


# ============================================================
# Backend API
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# Header
# ============================================================

st.title("🔬 LesionLens")

st.markdown(
    "AI-powered skin lesion classification with explainable predictions."
)

st.markdown("---")


# ============================================================
# Medical Disclaimer
# ============================================================

st.warning(
    "⚠️ **Educational tool only — not a medical diagnosis.** "
    "Always consult a dermatologist for any skin concern."
)


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Process Uploaded Image
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Create two columns
    # Left  → uploaded image
    # Right → prediction result
    col1, col2 = st.columns(2)


    # ========================================================
    # Display Uploaded Image
    # ========================================================

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


    # ========================================================
    # Analyze Button
    # ========================================================

    if st.button(
        "🔍 Analyze Lesion",
        type="primary"
    ):

        with st.spinner("Analyzing image..."):

            try:

                # ------------------------------------------------
                # Send Image to FastAPI Backend
                # ------------------------------------------------

                # Reset file pointer before sending
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )


                # =================================================
                # Successful Response
                # =================================================

                if response.status_code == 200:

                    result = response.json()


                    # =============================================
                    # Prediction Result
                    # =============================================

                    with col2:

                        st.subheader("Prediction Result")

                        st.markdown(
                            "**Predicted Condition:**"
                        )

                        st.markdown(
                            f"### {result['predicted_label']}"
                        )

                        st.markdown(
                            f"**Confidence:** "
                            f"{result['confidence']}%"
                        )

                        # Visual confidence indicator
                        st.progress(
                            result["confidence"] / 100
                        )


                    # =============================================
                    # Probability Breakdown
                    # =============================================

                    st.markdown(
                        "### Full Probability Breakdown"
                    )

                    probs_df = pd.DataFrame(
                        result["all_probabilities"].items(),
                        columns=[
                            "Class",
                            "Probability (%)"
                        ]
                    )


                    # Create interactive Plotly bar chart
                    fig = px.bar(
                        probs_df,
                        x="Class",
                        y="Probability (%)",
                        color="Probability (%)",
                        color_continuous_scale="Blues",
                        text="Probability (%)"
                    )

                    # Keep Y-axis between 0 and 100%
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        showlegend=False
                    )

                    # Display percentage values above bars
                    fig.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                    # =============================================
                    # Borderline Prediction Warning
                    # =============================================

                    top_two = list(
                        result["all_probabilities"].values()
                    )[:2]

                    if (
                        len(top_two) > 1
                        and (top_two[0] - top_two[1]) < 15
                    ):
                        st.info(
                            "ℹ️ This prediction is borderline — "
                            "the model was close between two possible "
                            "classes. Consider this result with extra caution."
                        )


                # =================================================
                # API Error
                # =================================================

                else:

                    st.error(
                        f"API Error: {response.status_code} — "
                        f"{response.text}"
                    )


            # =====================================================
            # Backend Connection Error
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the backend API. "
                    "Make sure the FastAPI server is running at "
                    "http://127.0.0.1:8000"
                )


            # =====================================================
            # Request Timeout
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The backend took too long to respond. "
                    "Please try again."
                )


            # =====================================================
            # Unexpected Error
            # =====================================================

            except Exception as error:

                st.error(
                    f"❌ Something went wrong: {error}"
                )