# ============================================================
# LesionLens - FastAPI Backend
# ============================================================
# This file creates the API for LesionLens.
#
# Main responsibilities:
# 1. Start the FastAPI application
# 2. Accept an uploaded skin-lesion image
# 3. Validate the uploaded file
# 4. Send the image to the trained model
# 5. Return the prediction to the frontend
# ============================================================


# ============================================================
# Imports
# ============================================================

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_utils import predict


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
# CORS allows our Streamlit frontend to communicate
# with this FastAPI backend when they run on different ports.
#
# Example:
# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000
#
# "*" is acceptable for local development.
# It should be restricted to the frontend URL in production.

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
    Simple endpoint to check whether the API is running.
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
    Receives an image from the frontend,
    runs the trained model,
    and returns the prediction.
    """

    # --------------------------------------------------------
    # Step 1: Validate File Type
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )


    # --------------------------------------------------------
    # Step 2: Read Uploaded Image
    # --------------------------------------------------------

    try:
        # Read the uploaded file as bytes
        image_bytes = await file.read()

        # Convert bytes into a PIL image
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
    # Step 4: Return Prediction to Frontend
    # --------------------------------------------------------

    return result