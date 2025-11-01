# Digits Classification using CNN

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/competitions/digit-recognizer)

This repository contains the code for a Convolutional Neural Network (CNN) model designed to classify handwritten digits (0-9). The project uses the classic **MNIST (Digit Recognizer)** dataset from Kaggle.

This project demonstrates a complete data science pipeline:
1.  **Data Loading & Exploration:** Importing the dataset and visualizing its properties.
2.  **Data Preprocessing:** Normalizing, reshaping, and encoding the data for the CNN.
3.  **Model Building:** Defining and compiling a `tensorflow.keras` Sequential CNN model.
4.  **Training & Evaluation:** Fitting the model and evaluating its performance on a validation set.
5.  **Prediction:** Using the trained model to make predictions on the unseen test set.

---

## Dataset

The model is trained on the [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) dataset from Kaggle.

* **`train.csv`**: Contains 42,000 labeled images. Each row has 785 values: the first value is the `label` (the digit), and the remaining 784 values are the pixel intensities (0-255) for the 28x28 image.
* **`test.csv`**: Contains 28,000 unlabeled images, formatted identically to the training set (minus the label column).

### Data Distribution

The training data is well-balanced across all 10 classes (digits 0-9), as shown in the count plot from the notebook.

![Data Distribution](https://i.imgur.com/L13sJt8.png)

---

## Methodology

The entire workflow is contained in the `digits-classification-using-cnn.ipynb` notebook. The key steps are as follows:

### 1. Preprocessing

Before being fed into the neural network, the data undergoes several crucial preprocessing steps:

1.  **Normalization:** Pixel values are scaled from the `[0, 255]` range to a `[0, 1]` range. This is achieved by dividing the pixel data by `255.0`. This helps the optimizer converge more efficiently.
2.  **Reshaping:** The flat 784-pixel vector for each image is reshaped into a `(28, 28, 1)` tensor. This 3D shape (height, width, channels) is required by the `Conv2D` layers, which operate on spatial data.
3.  **One-Hot Encoding:** The target labels (e.g., `5` or `7`) are converted into categorical (one-hot) vectors (e.g., `[0,0,0,0,0,1,0,0,0,0]`). This is necessary for using the `categorical_crossentropy` loss function.
4.  **Train-Validation Split:** The training data is split into training (80%) and validation (20%) sets to monitor the model's performance on unseen data during training and prevent overfitting.

### 2. Model Architecture

A Sequential CNN model is built using `tensorflow.keras`. The architecture is designed to capture spatial features from the images effectively.

The model consists of two convolutional blocks followed by a fully connected classification head:

1.  **Conv Block 1:** `Conv2D` (32 filters, 3x3 kernel, 'relu' activation) -> `MaxPool2D` (2x2)
2.  **Conv Block 2:** `Conv2D` (64 filters, 3x3 kernel, 'relu' activation) -> `MaxPool2D` (2x2)
3.  **Flatten:** Flattens the 2D feature maps into a 1D vector.
4.  **Classifier Head:**
    * `Dense` (128 units, 'relu' activation)
    * `Dropout` (0.2)
    * `Dense` (64 units, 'relu' activation)
    * `Dropout` (0.2)
    * `Dense` (10 units, 'softmax' activation) -> Output layer

The model is compiled using:
* **Optimizer:** `adam`
* **Loss:** `categorical_crossentropy`
* **Metrics:** `accuracy`

Here is the model summary:
