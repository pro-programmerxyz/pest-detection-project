import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model("C:/Users/Anant/Desktop/New folder/mobilenet_fine_tuned_model.h5")  


img_size = (224, 224)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

input_image_path = "C:/Users/Anant/Desktop/New folder/images.jpeg"


input_img_array = preprocess_image(input_image_path)


prediction = model.predict(input_img_array)
predicted_class_index = np.argmax(prediction[0])

class_labels = ['paddy with pest', 'paddy without pest']  
predicted_class = class_labels[predicted_class_index]

print(f"Input Image: {input_image_path}")
print(f"Predicted Class: {predicted_class}")
