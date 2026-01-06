import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model("plant_disease_model.keras")

# List of class names in the same order as your training generator
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

# Paths to your test images
img_paths = [
    r"E:\Emerging\processed\test\Potato___Early_blight\image (63).JPG",
    r"E:\Emerging\processed\test\Tomato___Tomato_mosaic_virus\image (9).JPG",
    r"E:\Emerging\processed\test\Tomato___Target_Spot\image (1).JPG",
    r"E:\Emerging\processed\test\Tomato___Tomato_Yellow_Leaf_Curl_Virus\image (167).JPG",
    r"E:\Emerging\processed\test\Tomato___healthy\image (2).JPG"
]

for img_path in img_paths:
    # Check if the image exists
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(160, 160))  # match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Make prediction
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]

    # Print result
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted class index: {pred_class}")
    print(f"Predicted label: {class_labels[pred_class]}")
    print("-" * 20)
