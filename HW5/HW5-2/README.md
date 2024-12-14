# MNIST Classification Homework

This homework involves building and training two different neural network models (DNN and CNN) to classify handwritten digits from the MNIST dataset. The models are implemented using TensorFlow and Keras.

## Files

- `DNN.ipynb`: Jupyter Notebook for building and training a Deep Neural Network (DNN) model.
- `CNN.ipynb`: Jupyter Notebook for building and training a Convolutional Neural Network (CNN) model.

## Requirements

- Python 3.10
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. Install the required packages:

## DNN Model

The `DNN.ipynb` notebook contains the following steps:

Import Required Libraries: Import TensorFlow, Keras, and NumPy.
Load and Preprocess the MNIST Dataset: Load the dataset, normalize the images, and convert the labels to categorical format.
Build the DNN Model: Define a sequential model with multiple dense layers.
Compile the Model: Compile the model with an appropriate optimizer, loss function, and evaluation metric.
Train the Model: Train the model using the training data and validate it using the validation data.
Evaluate the Model: Evaluate the model's performance on the test data and print the accuracy.
Make Predictions: Use the trained model to make predictions on new data and visualize the results.

## CNN Model

The `CNN.ipynb` notebook contains the following steps:

Import Required Libraries: Import TensorFlow, Keras, and other dependencies.
Load and Preprocess the MNIST Dataset: Load the dataset, normalize the pixel values, and reshape the input data.
Build the CNN Model: Define the CNN architecture using Keras, including convolutional layers, pooling layers, and dense layers.
Compile the Model: Compile the model with an appropriate optimizer, loss function, and evaluation metrics.
Train the Model: Train the CNN model on the training data and validate it on the validation data.
Evaluate the Model: Evaluate the trained model on the test data and report the accuracy and other metrics.
Make Predictions: Use the trained model to make predictions on new data and visualize the results.

## Running the Notebooks

1. Open the Jupyter Notebook server:
   ```bash
   jupyter notebook

2. Open DNN.ipynb and CNN.ipynb notebooks and run the cells sequentially to train and evaluate the models.

## Results
- The DNN model achieves a test accuracy of approximately 97.68%.
- The CNN model achieves a test accuracy of approximately 99.28%.

## Conclusion

Both models perform well on the MNIST dataset, with the CNN model achieving higher accuracy due to its ability to capture spatial features in the images.