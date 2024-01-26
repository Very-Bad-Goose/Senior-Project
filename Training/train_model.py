# Code for model training
# (Assuming you have a labeled dataset and TensorFlow installed)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and preprocess training and validation data
train_generator = datagen.flow_from_directory(
    'path/to/training_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path/to/training_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained MobileNet model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
                                               input_shape=(224, 224, 3),
                                               include_top=False)

# Add custom layers for binary classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=5)
