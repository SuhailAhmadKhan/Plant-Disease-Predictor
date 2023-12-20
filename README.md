# Plant Disease Predictor


## Overview

This project is a Plant Disease Predictor that utilizes deep learning to identify diseases in plant images. Users can upload images of plant leaves, and the system will provide predictions about potential diseases affecting the plants.

The model used for predictions is based on a pre-trained VGG19 neural network, fine-tuned for plant disease classification. The web application is built using Flask for the backend and Streamlit for the frontend.

## Features

- **Plant Disease Prediction**: Users can upload images, and the system will predict the likelihood of diseases affecting the plant.
- **User-Friendly Interface**: The web interface is designed to be intuitive and easy to use.

## Deployment

The Plant Disease Predictor is deployed and accessible at the following link: [Plant Disease Predictor App](https://plant-disease-predictor-u5xngdrrajinmvhnwdxay6.streamlit.app/)

## Getting Started

To run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
3. Run the Streamlit app:
    ```bash
    streamlit run app_streamlit.py

## Project Structure

- **static**: Contains static files such as uploaded images.
- **templates**: HTML templates for the Flask web application.
- **app.py**: Flask web application for local deployment.
- **app_streamlit.py**: Streamlit web application for deployment on Streamlit sharing.
- **best_model.h5**: Pre-trained deep learning model for plant disease prediction.
- **requirements.txt**: List of dependencies required for the project.

## Project Lifecycle

1. **Planning and Research**
   - Identified the problem: Plant Disease Detection.
   - Researched existing solutions and technologies.

2. **Data Collection**
   - Gathered a diverse dataset of plant images with disease labels.
   - Split the dataset into training and testing sets.

3. **Model Development**
   - Chose VGG19 architecture for deep learning.
   - Developed and trained the model using TensorFlow and Keras.

4. **Web Application Development**
   - Created a Flask web application for local deployment (app.py).
   - Deployed a Streamlit web application for online sharing (app_streamlit.py).

5. **Testing and Validation**
   - Tested the model with various plant images.
   - Validated predictions against ground truth labels.

6. **Optimization**
   - Fine-tuned the model for better accuracy.
   - Optimized code and resources for efficient deployment.

7. **Deployment**
   - Deployed the Flask application locally for testing.
   - Deployed the Streamlit application on Streamlit Sharing.

8. **Documentation**
   - Created comprehensive documentation for code and usage.
   - Generated a README file for project information.

9. **Maintenance**
   - Regularly updated dependencies and addressed issues.
   - Monitored and improved prediction accuracy over time.

10. **Public Release**
    - Published the project on GitHub for open-source access.
    - Shared the Streamlit deployment link: [Plant Disease Predictor](https://plant-disease-predictor-u5xngdrrajinmvhnwdxay6.streamlit.app/).


