# Iris Classification with TensorBoard

![web](/HW5/HW5-1/images/screenshot.png)

This homework demonstrates how to solve the Iris classification problem using TensorFlow and visualize the training process with TensorBoard.

## Overview

The Iris dataset is a classic dataset used for machine learning and contains three classes of iris plants: Setosa, Versicolor, and Virginica. This project involves training a neural network to classify iris plants based on their features and using TensorBoard to visualize the training process.

## Requirements

- Python 3.x
- TensorFlow
- TensorBoard
- scikit-learn
- numpy

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/YOUR_USERNAME/iris-tensorboard.git
    cd iris-tensorboard
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Open the Jupyter Notebook**:
    ```sh
    jupyter notebook Iris_tensorboard.ipynb
    ```

2. **Run the Notebook**:
    Follow the instructions in the notebook to load the Iris dataset, preprocess the data, build and train the model, and visualize the training process using TensorBoard.

3. **Start TensorBoard**:
    After running the notebook, start TensorBoard to visualize the training logs:
    ```sh
    tensorboard --logdir=logs/fit
    ```

4. **Open TensorBoard in a Browser**:
    Open your web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

## File Structure

- `Iris_tensorboard.ipynb`: Jupyter Notebook containing the code to solve the Iris classification problem and visualize the training process with TensorBoard.
- `requirements.txt`: List of required Python packages.

## Code Explanation

### Load and Preprocess Data

The Iris dataset is loaded and split into training and testing sets. The features are standardized using `StandardScaler`.

### Build and Compile Model

A neural network model is built using Keras and compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.

### Train Model with TensorBoard

The model is trained using the training data, and a TensorBoard callback is used to log the training process. Custom summaries such as scalar values, histograms, and images are also logged.

### Evaluate Model

The model is evaluated on the test set to determine its accuracy.

### Visualize with TensorBoard

TensorBoard is used to visualize the training process, including loss and accuracy curves, weight histograms, and sample images.

## Example Output

After running the notebook and starting TensorBoard, you should see visualizations like the following:

- **Scalar Dashboard**: Visualizes scalar statistics such as loss and accuracy over time.
- **Histogram Dashboard**: Displays the distribution of weights and biases over time.
- **Image Dashboard**: Shows sample images from the dataset.

## License

This project is licensed under the MIT License.