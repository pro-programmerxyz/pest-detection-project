import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.random.set_seed(42)
data_dir = "C:/Users/Anant/Desktop/New folder/train"

img_size = (224, 224)  
batch_size = 32
num_classes = 2


datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2) 
train_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                              class_mode='categorical', subset='training', shuffle=True)
val_generator = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                            class_mode='categorical', subset='validation', shuffle=False)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  
base_model.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


epochs_fine_tuning = 10  
model.fit(train_generator, epochs=epochs_fine_tuning, validation_data=val_generator)

model.save("mobilenet_fine_tuned_model.h5")
