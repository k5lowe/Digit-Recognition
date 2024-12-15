# Digit-Recognition
A neural network-based digit recognition system using the MNIST dataset to classify handwritten digits with preprocessing and visualization (Coursera Project).

## Description ##

This Coursera project is a neural network-based digit recognition system that utilizes the MNIST dataset to classify handwritten digits (0–9). The dataset comprises 60,000 training and 10,000 testing images of 28x28 grayscale pixels, each representing a single digit. The project includes data preprocessing, model design, training, evaluation, and visualization, providing a comprehensive pipeline for solving this classification problem.

Key Metrics:
- Accuracy: 97.50%
- Loss: ~0.14

Features:
- Classifies digits from 0 to 9 using the MNIST dataset
- Data Preprocessing:
  - Flattens 3D image data into 2D arrays
  - Normalizes pixel values to the range [-1, 1]
- Neural Network Model
- Model Training
  - Trains the neural network using the Adam optimizer
  - Utilizes Sparse Categorical Crossentropy loss function
- Calculates model loss and accuracy on the test dataset
- Prediction Visualization
  
Technologies:
- Python
- TensorFlow/Keras
- MNIST Dataset
- Matplotlib
- NumPy

## Graphs ##

![image](https://github.com/user-attachments/assets/28f30537-6666-4a90-bc7a-373d037aa2f6)


## Attributions & Remarks ##
The entirity of this project rested on the kind help of the following websites, especially Coursera and analyticsvidhya. As always, ChatGPT proved an immense help and guidance where much time and effort was saved.

- https://chatgpt.com/
- https://www.coursera.org/
- https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/
- https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/
- http://neuralnetworksanddeeplearning.com/chap1.html
- https://www.tutorialspoint.com/
- https://www.kaggle.com/code/mikelkn/mnist-prediction
- https://saturncloud.io/blog/how-to-improve-accuracy-in-neural-networks-with-keras/

Please feel free to use this code! Lastly, any feedback on how to better this code would be greatly appreciated. Thank you for your time.
