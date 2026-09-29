# ============================================================
# LesionLens - FastAPI Backend
# ============================================================
# This file creates the backend API for LesionLens.
#
# Main responsibilities:
# 1. Start the FastAPI application
# 2. Receive uploaded skin-lesion images
# 3. Run EfficientNet-B0 prediction
# 4. Generate Grad-CAM explanations
# 5. Return prediction data and Grad-CAM images
# ============================================================


# ============================================================
# Imports
# ============================================================

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from model_utils import (
    predict,
    generate_gradcam_overlay
)


# ============================================================
# Create FastAPI Application
# ============================================================

app = FastAPI(
    title="LesionLens API",
    version="1.0"
)


# ============================================================
# CORS Configuration
# ============================================================
# CORS allows the Streamlit frontend to communicate with
# the FastAPI backend when both run on different ports.
#
# Local development:
# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000
#
# "*" is fine during local development.
# It should be restricted in production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    """
    Checks whether the FastAPI backend is running.
    """

    return {
        "message": "LesionLens API is running."
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
async def predict_lesion(
    file: UploadFile = File(...)
):
    """
    Receives a skin-lesion image and returns:
    - Predicted class
    - Human-readable label
    - Confidence
    - Probability of every class
    """

    # --------------------------------------------------------
    # Step 1: Validate File Type
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )


    # --------------------------------------------------------
    # Step 2: Read Uploaded Image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )


    # --------------------------------------------------------
    # Step 3: Run Model Prediction
    # --------------------------------------------------------

    result = predict(image)


    # --------------------------------------------------------
    # Step 4: Return Prediction
    # --------------------------------------------------------

    return result


# ============================================================
# Grad-CAM Endpoint
# ============================================================

@app.post("/gradcam")
async def gradcam_lesion(
    file: UploadFile = File(...)
):
    """
    Receives a skin-lesion image and returns
    the Grad-CAM visualization as a PNG image.
    """

    # --------------------------------------------------------
    # Step 1: Validate File Type
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )


    # --------------------------------------------------------
    # Step 2: Read Uploaded Image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )


    # --------------------------------------------------------
    # Step 3: Generate Grad-CAM
    # --------------------------------------------------------

    overlay_image, _ = generate_gradcam_overlay(
        image
    )


    # --------------------------------------------------------
    # Step 4: Convert PIL Image to PNG Bytes
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    overlay_image.save(
        image_buffer,
        format="PNG"
    )

    # Move pointer back to the beginning
    image_buffer.seek(0)


    # --------------------------------------------------------
    # Step 5: Return Grad-CAM Image
    # --------------------------------------------------------

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )