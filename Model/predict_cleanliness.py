# Code for image recognition
# (Assuming you have a trained model and OpenCV installed)

import cv2
import numpy as np

def predict_cleanliness(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values

    score = model.predict(img)
    return score[0][0]
