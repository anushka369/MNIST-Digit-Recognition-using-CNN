# MNIST Digit Recognition using CNN 🧠

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) from the **MNIST dataset**. The model achieves high accuracy (99.20%) and demonstrates the power of deep learning for image classification.

---

## 📌 **Project Overview**  
- Uses **Keras & TensorFlow** to build and train a CNN model.  
- The MNIST dataset contains **60,000 training** and **10,000 test images** (28×28 pixels, grayscale).  
- Achieves **99.20% test accuracy** using **Conv2D, MaxPooling, ReLU, Softmax, and Adam optimizer**.  

---

## 🔧 **Installation & Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/mnist-cnn.git
   cd mnist-cnn
   ```
   
2. Install dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib
   ```

3. Run the training script:
   ```bash
   python mnist_cnn.py
   ```

---

## 📂 **Project Structure**

  ```bash
  📦 mnist-cnn  
   ┣ 📜 mnist_cnn.py            # Main training script  
   ┣ 📜 model.keras             # Saved model in Keras format  
   ┣ 📜 README.md               # Project documentation  
   ┣ 📜 requirements.txt        # Dependencies  
   ┣ 📜 test_sample.png         # Example test image
  ```

---

## 📊 **Model Architecture**

```
--------------------------------------------------------------------------------
| Layer      | Type            | Filters | Kernel | Activation | Output Shape  |
--------------------------------------------------------------------------------
| Conv2D     | Convolution     |   32    |  3×3   |    ReLU    | (26, 26, 32)  |
| MaxPooling | Pooling         |    -    |  2×2   |     -      | (13, 13, 32)  |
| Conv2D     | Convolution     |   64    |  3×3   |    ReLU    | (11, 11, 64)  |
| MaxPooling | Pooling         |    -    |  2×2   |     -      |  (5, 5, 64)   |
| Flatten    | Fully Connected |    -    |   -    |     -      |    (1600)     |
| Dense      | Fully Connected |   128   |   -    |    ReLU    |    (128)      |
| Dense      | Fully Connected |    10   |   -    |  Softmax   |    (10)       |
--------------------------------------------------------------------------------
```

---

## 🎯 Results & Performance

✅ Training Accuracy: 99.50% <br>
✅ Test Accuracy: 99.20% <br>
✅ Loss Function: Categorical Cross-Entropy <br>
✅ Optimizer: Adam

---

## 🎨 **Sample Predictions**

The examples of digit predictions made by the model are provided in `test_sample.png`.

---

## 🚀 **How to Use the Model**

1️⃣ **Predict on New Images**
You can load the trained model and use it to classify new handwritten digits:

``` python
from tensorflow import keras
import numpy as np
import cv2

# Load the model
model = keras.models.load_model("model.keras")

# Load and preprocess an image
img = cv2.imread("test_sample.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28)) / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for CNN

# Predict
prediction = model.predict(img)
print("Predicted Digit:", np.argmax(prediction))
```

---

## 🔮 **Future Improvements**

- Enhance Model: Try deeper CNNs like VGG16, ResNet, or transfer learning.
- Data Augmentation: Improve generalization by rotating, flipping, and adding noise.
- Deploy Model: Convert to TensorFlow Lite for mobile apps or use Flask/FastAPI for web deployment.

---

## 🤖 **Tech Stack**

- Language: Python
- Libraries: TensorFlow, Keras, NumPy, OpenCV, Matplotlib
- Model Type: Convolutional Neural Network (CNN)

---

## 🩹 **Contributions**

This is a personal learning project, but submitting issues and suggestions are welcome!
<br> If you find any improvements, feel free to create a pull request. To contribute:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and submit a pull request.

---

## ⭐ **Author**

Developed by **Anushka**. <br>
📧 [ab8991@srmist.edu.in](mailto:ab8991@srmist.edu.in)

---
