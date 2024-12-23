# Homework 5

This repository contains three assignments for the class, focusing on different machine learning tasks using various frameworks and libraries.

### HW5-1: Iris Classification Problem

This assignment involves solving the Iris classification problem using TensorFlow (tf.keras), PyTorch, and PyTorch Lightning. The goal is to classify iris plants into three species based on their features and visualize the training process using TensorBoard.

#### Files
- `IRIS_TENSORBOARD.ipynb`: Jupyter Notebook for solving the Iris classification problem using TensorFlow and visualizing the training process with TensorBoard.
- `requirements.txt`: List of required Python packages.

#### Instructions
1. **Open the Jupyter Notebook**:
    ```sh
    jupyter notebook IRIS_TENSORBOARD.ipynb
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

### HW5-2: Handwritten Digit Recognition

This assignment involves building and training two different neural network models (Dense Neural Network and Convolutional Neural Network) to classify handwritten digits from the MNIST dataset using TensorFlow (tf.keras).

#### Files
- `DNN.ipynb`: Jupyter Notebook for building and training a Dense Neural Network (DNN) model.
- `CNN.ipynb`: Jupyter Notebook for building and training a Convolutional Neural Network (CNN) model.
- `requirements.txt`: List of required Python packages.

#### Instructions
1. **Open the Jupyter Notebook server**:
    ```sh
    jupyter notebook
    ```

2. **Open `DNN.ipynb` and `CNN.ipynb` notebooks**:
    Run the cells sequentially to train and evaluate the models.

#### Results
- The DNN model achieves a test accuracy of approximately 97.68%.
- The CNN model achieves a test accuracy of approximately 99.28%.

### HW5-3: CIFAR Image Classification

This assignment involves classifying images from the CIFAR-10 dataset using VGG16 and VGG19 architectures, either pretrained or trained from scratch, using TensorFlow (tf.keras) or PyTorch Lightning.

#### Files
- `CIFAR_VGG16.ipynb`: Jupyter Notebook for building and training a VGG16 model on the CIFAR-10 dataset.
- `requirements.txt`: List of required Python packages.

#### Instructions
1. **Open the Jupyter Notebook**:
    ```sh
    jupyter notebook CIFAR_VGG16.ipynb
    ```

2. **Run the Notebook**:
    Follow the instructions in the notebook to load the CIFAR-10 dataset, preprocess the data, build and train the VGG16 model, and evaluate its performance.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License.