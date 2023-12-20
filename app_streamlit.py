import streamlit as st
import cv2
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

model = load_model('best_model.h5')

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(img):
    ref = {0: 'Apple___Apple_scab',1: 'Apple___Black_rot',2: 'Apple___Cedar_apple_rust',3: 'Apple___healthy',4: 'Blueberry___healthy',5: 'Cherry_(including_sour)___Powdery_mildew',6: 'Cherry_(including_sour)___healthy',7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',8: 'Corn_(maize)___Common_rust_',9: 'Corn_(maize)___Northern_Leaf_Blight',10: 'Corn_(maize)___healthy',11: 'Grape___Black_rot',12: 'Grape___Esca_(Black_Measles)',13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',14: 'Grape___healthy',15: 'Orange___Haunglongbing_(Citrus_greening)',16: 'Peach___Bacterial_spot',17: 'Peach___healthy',18: 'Pepper,_bell___Bacterial_spot',19: 'Pepper,_bell___healthy',20: 'Potato___Early_blight',21: 'Potato___Late_blight',22: 'Potato___healthy',23: 'Raspberry___healthy',24: 'Soybean___healthy',25: 'Squash___Powdery_mildew',26: 'Strawberry___Leaf_scorch',27: 'Strawberry___healthy',28: 'Tomato___Bacterial_spot',29: 'Tomato___Early_blight',30: 'Tomato___Late_blight',31: 'Tomato___Leaf_Mold',32: 'Tomato___Septoria_leaf_spot',33: 'Tomato___Spider_mites Two-spotted_spider_mite',34: 'Tomato___Target_Spot',35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',36: 'Tomato___Tomato_mosaic_virus',37: 'Tomato___healthy'}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))  # Resize image
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    pred = np.argmax(model.predict(img))
    return ref[pred]

st.title('Plant Disease Recognition')

uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make a prediction
    result = prediction(image)
    st.write(f"Prediction: {result}")
