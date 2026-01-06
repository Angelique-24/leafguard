import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from scipy import ndimage

# ==============================
# Class Labels
# ==============================
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    

    "Tomato___Tomato_mosaic_virus",
     
    "Tomato___healthy"
    
]

# ==============================
# Sample Images
# ==============================
sample_images = {
    "Healthy Apple": "processed/test/Apple___healthy/image (1).JPG",
    "Apple Scab": "processed/test/Apple___Apple_scab/image (112).JPG",
    "Healthy Tomato": "processed/test/Tomato___healthy/image (101).JPG",
    "Tomato Late Blight": "processed/test/Tomato___Late_blight/image (1010).JPG"
}

# ==============================
# Load Trained Model
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ==============================
# Prediction Function
# ==============================
def run_prediction(img):
    # --- Leaf Detection ---
    gray_img = np.array(img.convert('L'))
    blurred = ndimage.gaussian_filter(gray_img, sigma=2)
    mask = blurred < 200
    labeled_mask, num_labels = ndimage.label(mask)

    if num_labels > 0:
        coords = np.where(labeled_mask == 1)
        y1, y2 = np.min(coords[0]), np.max(coords[0])
        x1, x2 = np.min(coords[1]), np.max(coords[1])
        leaf_crop = img.crop((x1, y1, x2, y2))
    else:
        leaf_crop = img

    processed_img = np.expand_dims(image.img_to_array(leaf_crop.resize((160, 160))), axis=0)

    # Predict
    prediction = model.predict(processed_img)
    pred_class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    label = class_labels[pred_class_idx]
    plant, status = label.split("___")
    
    return {"plant": plant, "disease": status, "confidence": confidence}


def display_prediction_result(img, result):
    st.subheader("🔎 Prediction Result")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Input Image", width=300)
    with col2:
        st.write(f"**Plant:** {result['plant'].replace('_', ' ')}")
        st.write(f"**Disease:** {result['disease'].replace('_', ' ')}")
        st.write(f"**Confidence:** {result['confidence']:.2f}%")

# ==============================
# Information Page
# ==============================
def information_page():
    st.title("🌱 LEAFGUARD: PLANT DISEASE DETECTION USING CONVOLUTIONAL NEURAL NETWORKS")

    # Show prediction if a sample was clicked
    if 'predict_image_path' in st.session_state:
        img_path = st.session_state.predict_image_path
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            result = run_prediction(img)
            display_prediction_result(img, result)
            
            if st.button("⬅️ Back to Home"):
                del st.session_state.predict_image_path
                st.rerun()
        else:
            st.error("Image not found.")
            del st.session_state.predict_image_path

    else:
        st.write(
            "This system helps you diagnose plant diseases from leaf images. "
            "It uses a deep learning model to classify a single leaf image at a time. "
            "Try it out with the sample images below, or upload your own."
        )

        if st.button("⬆️ Upload Your Own Image", key="upload_page_button"):
            st.session_state.page = "upload"
            st.rerun()

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        # --- Sample Images ---
        st.subheader("Try with a sample image")
        
        cols = st.columns(len(sample_images))
        for idx, (caption, path) in enumerate(sample_images.items()):
            with cols[idx]:
                if os.path.exists(path):
                    st.image(path, caption=caption, width=150)
                    if st.button(f"Predict {caption}", key=f"predict_{caption}"):
                        st.session_state.predict_image_path = path
                        st.rerun()
                else:
                    st.warning(f"Sample image not found:\n{path}")
        
        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        # Supported Plants section
        st.subheader("Supported Plants and Diseases")
        plant_diseases = {}
        for name in class_labels:
            if "___" not in name:
                continue
            plant, disease = name.split("___")
            if plant not in plant_diseases:
                plant_diseases[plant] = []
            plant_diseases[plant].append(disease.replace("_", " "))

        sorted_plants = sorted(plant_diseases.items())
        
        cols = st.columns(4)
        for i, (plant, diseases) in enumerate(sorted_plants):
            with cols[i % 4]:
                st.markdown(f"**{plant}**")
                st.markdown("- " + "\n- ".join(sorted(diseases)))


# ==============================
# Upload Page
# ==============================
def upload_page():
    st.title("Upload and Predict")

    if st.button("⬅️ Go Back"):
        st.session_state.page = "info"
        if 'predict_image_path' in st.session_state:
            del st.session_state.predict_image_path
        st.rerun()

    # --- Uploader ---
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        result = run_prediction(img)
        display_prediction_result(img, result)

    st.info(
        "Disclaimer: This is an AI-powered tool and may not be 100% accurate. "
        "Always consult with a qualified expert for a definitive diagnosis."
    )

# ==============================
# Main App Logic
# ==============================
st.set_page_config(page_title="Plant Disease Diagnosis", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "info"
    
if "predict_image_path" in st.session_state and st.session_state.page == "upload":
    del st.session_state.predict_image_path

if st.session_state.page == "info":
    information_page()
else:
    upload_page()

