import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

#   Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

#   Normalize image data
training_images, testing_images = training_images / 255.0, testing_images / 255.0

#   Class names for CIFAR-10 
# I have Checked only two ( Deer and Dog )
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#   Load the trained model
model_path = "image_classifier.keras" 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found! Check the path.")

model = models.load_model(model_path)
print(" Model loaded successfully!")

#   Define image path 
# It will be predicted as DOG and DEER respectively depending upon the path you provide. 
img_path = r"C:\Users\Muhammad Ibtesam\Desktop\AI Project\deer.jpg"   


#   Check if the file exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f" Image file '{img_path}' not found! Check the file name and extension.")

#   Load and preprocess the test image
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)  # Resize for model compatibility
img = img / 255.0  # Normalize pixel values

#   Expand dimensions to match model input
img = np.expand_dims(img, axis=0)

#   Make a prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

#   Display the result
print(f" Prediction: {class_names[predicted_class]}")
