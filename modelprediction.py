import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("C:/Users/Anant/Desktop/New folder/mobilenet_fine_tuned_model.h5")  # Replace with the path to your saved model file

# Define image size (should be the same as used during training)
img_size = (224, 224)

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Define the path to your single input image
input_image_path = "C:/Users/Anant/Desktop/New folder/images.jpeg"

# Preprocess the input image
input_img_array = preprocess_image(input_image_path)

# Make prediction for the input image
prediction = model.predict(input_img_array)
predicted_class_index = np.argmax(prediction[0])

class_labels = ['paddy with pest', 'paddy without pest']  # Replace with your class labels
predicted_class = class_labels[predicted_class_index]

print(f"Input Image: {input_image_path}")
print(f"Predicted Class: {predicted_class}")
