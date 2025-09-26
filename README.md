# Formula-One-Cars-AI-Image-Detection
An AI-powered project for detecting and recognizing Formula One cars in images using computer vision and deep learning techniques. This project aims to identify F1 cars, their teams, and optionally drivers, providing insights for analytics, fan applications, and automated media processing.
Link download: https://drive.google.com/drive/folders/1TuEf36SDW5iRasMRlifSL9Xqid1-uhlU?usp=drive_link 
Formula One Car Classifier 🏎️
This repository contains the code for a Convolutional Neural Network (CNN) model trained to classify images of Formula One cars from different teams. The model was trained using Google Colab with a custom dataset.

1. Overview
The goal of this project is to build an image classification model that can identify the F1 team of a car from its image. The model is built using TensorFlow and Keras and is trained on a custom dataset of F1 cars.

2. Getting Started
To run this project, you'll need a Google account and access to Google Colab, as the training process leverages its free GPU resources.

2.1. Clone the Repository
Start by cloning this repository to your local machine or directly into Google Colab.

Bash

!git clone https://github.com/your-username/your-repo-name.git
Replace https://github.com/your-username/your-repo-name.git with the actual URL of your repository.

2.2. Upload Your Dataset
Your dataset, named dataset.zip, should be a compressed file containing the images. Upload this file to your Google Drive, ideally within a dedicated folder like AI Project.

3. Project Structure
The project has a straightforward structure designed for clarity and ease of use.

/
├── README.md
├── formula_one_car_classifier.h5 (Your trained model file)
└── training_notebook.ipynb (Your Colab Notebook)
4. Training the Model
The training process is detailed in the training_notebook.ipynb file. You can open this file directly in Google Colab to run the code.

Follow these steps within the Notebook:

Connect to a GPU Runtime: Go to Runtime -> Change runtime type and select GPU under Hardware accelerator.

Mount Google Drive: Run the following code cell to connect your Google Drive, allowing you to access your dataset.



from google.colab import drive
drive.mount('/content/drive')
Unzip the Dataset: Your dataset.zip file needs to be unzipped before use. The code below will extract it to the Colab environment.



zip_path = '/content/drive/MyDrive/AI Project/dataset.zip'
!unzip -qq "$zip_path" -d /content/
Data Preparation: The code will then automatically process the unzipped files and split them into training and validation sets, arranging them in the directory structure required by Keras.

Train the Model: The notebook will define and compile the CNN model, then start the training process. This may take some time depending on your dataset size.

Save the Model: After training, the model is saved as formula_one_car_classifier.h5 to your Google Drive for future use.

5. Using the Trained Model for Prediction
You can use the saved .h5 file to make predictions on new images without retraining the model.

5.1. Load the Model
First, load the model into your Python environment.


import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = 'https://github.com/your-username/your-repo-name/raw/main/formula_one_car_classifier.h5'
model = load_model(tf.keras.utils.get_file('formula_one_car_classifier.h5', model_path))
Note: The get_file utility automatically handles downloading the model from the provided URL, making it easy to use the model directly from GitHub.

5.2. Make a Prediction
Prepare your new image and use the model to predict its class.



import numpy as np
from tensorflow.keras.preprocessing import image

# Path to your test image
test_image_path = 'path/to/your/test_image.jpg'

# Load and preprocess the image
img_height = 128
img_width = 128
img = image.load_img(test_image_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
processed_image = np.expand_dims(img_array, axis=0) / 255.0

# Define your class labels (must be in the same order as when you trained)
class_labels = ['Alpha', 'Ferrari', 'Mercedes', 'RedBull', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10']

# Get the prediction
predictions = model.predict(processed_image)
predicted_class_index = np.argmax(predictions, axis=1)[0]
confidence = predictions[0][predicted_class_index]
predicted_class_name = class_labels[predicted_class_index]

print(f"Prediction: {predicted_class_name}")
print(f"Confidence: {confidence*100:.2f}%")
