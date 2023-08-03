import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility (optional)
tf.random.set_seed(42)

# Define data directory
data_dir = "C:/Users/Anant/Desktop/New folder/train"

# Define image size and batch size
img_size = (224, 224)  # Input size for MobileNet
batch_size = 32
num_classes=2

# Create data generator
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)  # 80% for training, 20% for validation
train_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                              class_mode='categorical', subset='training', shuffle=True)
val_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                            class_mode='categorical', subset='validation', shuffle=False)

# Create MobileNet base model (without top layers)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for our specific task
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Replace 'num_classes' with the number of classes in your dataset

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # Adjust the number of epochs based on your dataset and computational resources
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the trained model
model.save("mobilenet_model.h5")
