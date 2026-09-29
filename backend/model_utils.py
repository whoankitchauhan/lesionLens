# ============================================================
# LesionLens - Model Utilities
# ============================================================
# This file is responsible for:
# 1. Loading the trained EfficientNet-B0 model
# 2. Loading the class mapping
# 3. Removing hair from uploaded images
# 4. Preprocessing images
# 5. Making predictions
# ============================================================


# ============================================================
# Imports
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# File Paths
# ============================================================

# Get the folder containing this file
BASE_DIR = Path(__file__).resolve().parent

# Path to the trained model
MODEL_PATH = (
    BASE_DIR.parent
    / "model"
    / "lesionlens_model.pth"
)

# Path to the class mapping
CLASS_MAP_PATH = (
    BASE_DIR.parent
    / "model"
    / "class_to_idx.json"
)


# ============================================================
# Device
# ============================================================

# Use GPU if available, otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Load Class Mapping
# ============================================================

# Load the class-to-number mapping saved during training
with open(CLASS_MAP_PATH, "r") as file:
    class_to_idx = json.load(file)


# Convert:
# {"akiec": 0, "bcc": 1, ...}
#
# into:
# {0: "akiec", 1: "bcc", ...}

idx_to_class = {
    int(index): class_name
    for class_name, index in class_to_idx.items()
}


# ============================================================
# Human-Readable Class Names
# ============================================================

# These names will be shown in the LesionLens UI

CLASS_INFO = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi (Common Mole)",
    "vasc": "Vascular Lesion",
}


# ============================================================
# Load EfficientNet-B0
# ============================================================

def load_model():
    """
    Creates the EfficientNet-B0 architecture
    and loads our trained weights.
    """

    # Create EfficientNet-B0 without ImageNet weights
    model = models.efficientnet_b0(
        weights=None
    )

    # Replace the final layer
    # with the 7 HAM10000 classes
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_to_idx)
    )

    # Load our trained model weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    # Move model to GPU/CPU
    model = model.to(device)

    # Put model into evaluation mode
    model.eval()

    return model


# Load the trained model once when the backend starts
model = load_model()


# ============================================================
# Hair Removal
# ============================================================

def remove_hair(image):
    """
    Removes dark hair-like structures from a skin image.
    This is the same preprocessing method used during training.
    """

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Create an elliptical kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    # Detect dark hair-like structures
    blackhat = cv2.morphologyEx(
        gray_image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Create a binary hair mask
    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Remove detected hair
    cleaned_image = cv2.inpaint(
        image,
        hair_mask,
        1,
        cv2.INPAINT_TELEA
    )

    return cleaned_image


# ============================================================
# Image Preprocessing
# ============================================================

# IMPORTANT:
# This must match the evaluation preprocessing
# used during model training.

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Prediction Function
# ============================================================

def predict(image: Image.Image):
    """
    Takes an uploaded PIL image and returns
    the model's prediction and probabilities.
    """

    # Make sure image is RGB
    image = image.convert("RGB")

    # Remove hair
    cleaned_image = Image.fromarray(
        remove_hair(image)
    )

    # Apply the same preprocessing used during training
    tensor = eval_transform(
        cleaned_image
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move image to GPU/CPU
    tensor = tensor.to(device)

    # Make prediction
    with torch.no_grad():

        outputs = model(tensor)

        # Convert model outputs into probabilities
        probabilities = torch.nn.functional.softmax(
            outputs,
            dim=1
        )[0]

    # Get class with highest probability
    pred_idx = int(
        torch.argmax(probabilities).item()
    )

    # Convert class number to class code
    pred_class = idx_to_class[pred_idx]


    # ========================================================
    # Probability of Every Class
    # ========================================================

    all_probabilities = {
        idx_to_class[i]: round(
            float(probabilities[i]) * 100,
            2
        )
        for i in range(len(probabilities))
    }

    # Sort from highest probability to lowest
    all_probabilities = dict(
        sorted(
            all_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


    # ========================================================
    # Return Prediction Result
    # ========================================================

    return {
        "predicted_class": pred_class,

        "predicted_label": CLASS_INFO[
            pred_class
        ],

        "confidence": round(
            float(probabilities[pred_idx]) * 100,
            2
        ),

        "all_probabilities": all_probabilities
    }
    
    
# ============================================================
# Grad-CAM Explainability
# ============================================================
# Grad-CAM (Gradient-weighted Class Activation Mapping)
# helps us visualize which parts of the image influenced
# the model's prediction.
#
# Red/yellow regions → stronger influence
# Blue regions       → weaker influence
# ============================================================

import torch.nn.functional as F
import matplotlib.cm as cm


# ============================================================
# Grad-CAM Class
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):
        """
        Connects Grad-CAM to a specific layer of the model.
        """

        self.model = model
        self.target_layer = target_layer

        # These will store information during forward/backward pass
        self.gradients = None
        self.activations = None

        # Save activations during the forward pass
        target_layer.register_forward_hook(
            self._save_activation
        )

        # Save gradients during the backward pass
        target_layer.register_full_backward_hook(
            self._save_gradient
        )


    # --------------------------------------------------------
    # Save Forward Activations
    # --------------------------------------------------------

    def _save_activation(self, module, input, output):

        self.activations = output.detach()


    # --------------------------------------------------------
    # Save Backward Gradients
    # --------------------------------------------------------

    def _save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    def generate(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for an input image.
        """

        # Make sure the model is in evaluation mode
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # If no class is provided, explain the predicted class
        if class_idx is None:
            class_idx = output.argmax(
                dim=1
            ).item()

        # Clear previous gradients
        self.model.zero_grad()

        # Backpropagate the selected class score
        output[0, class_idx].backward()

        # Calculate importance of each feature channel
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Combine activations using the calculated weights
        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Remove negative values
        cam = F.relu(cam)

        # Resize heatmap to our model input size
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # Convert tensor to NumPy array
        cam = cam.squeeze().cpu().numpy()

        # Normalize values between 0 and 1
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam, class_idx


# ============================================================
# Initialize Grad-CAM
# ============================================================

# Use the final feature layer of EfficientNet-B0.
# This is the same target layer used in the Kaggle notebook.

target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# Denormalize Image
# ============================================================

def denormalize(tensor):
    """
    Converts a normalized image tensor back
    into a normal image range of 0-1.
    """

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    # Convert from CHW → HWC
    image = tensor.cpu().numpy().transpose(
        1, 2, 0
    )

    # Undo ImageNet normalization
    image = std * image + mean

    return np.clip(
        image,
        0,
        1
    )


# ============================================================
# Generate Grad-CAM Overlay
# ============================================================

def generate_gradcam_overlay(image: Image.Image):
    """
    Takes a PIL image and returns:
    1. Grad-CAM overlay as a PIL image
    2. Predicted class code
    """

    # Make sure image is RGB
    image = image.convert("RGB")

    # Apply the same hair removal used during training
    cleaned_image = Image.fromarray(
        remove_hair(image)
    )

    # Apply the same preprocessing used for prediction
    tensor = eval_transform(
        cleaned_image
    ).unsqueeze(0).to(device)

    # Generate Grad-CAM
    cam, pred_idx = gradcam.generate(
        tensor
    )

    # Convert normalized tensor back to normal image
    original_image = denormalize(
        tensor.squeeze(0)
    )

    # Convert heatmap values into colors
    heatmap = cm.jet(cam)[:, :, :3]

    # Blend original image and heatmap
    overlay = (
        0.5 * original_image
        + 0.5 * heatmap
    )

    # Keep values between 0 and 1
    overlay = np.clip(
        overlay,
        0,
        1
    )

    # Convert NumPy array → PIL image
    overlay_image = Image.fromarray(
        (overlay * 255).astype(np.uint8)
    )

    return (
        overlay_image,
        idx_to_class[pred_idx]
    )