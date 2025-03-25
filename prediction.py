from tensorflow import keras
import numpy as np
import cv2

# Load the trained model
model = keras.models.load_model("model.keras")

# Load an image for testing (convert to grayscale and resize to 28x28)
img = cv2.imread("test_sample.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for CNN input

# Predict the digit
prediction = model.predict(img)
digit = np.argmax(prediction)

print(f"Predicted Digit: {digit}")
