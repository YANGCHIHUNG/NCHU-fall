# Homework 6

This repository contains two assignments for the class, focusing on different machine learning tasks using various frameworks and libraries.

## HW6-1: Mask Detection

This assignment involves developing a machine learning model to detect whether a person is wearing a mask or not using images. The model is built using TensorFlow and a pretrained VGG16 model for transfer learning.

### CRISP-DM Process

#### 1. Business Understanding

**Objective:** Develop a machine learning model to detect whether a person is wearing a mask or not using images. This model can be used in various applications such as ensuring compliance with mask-wearing policies in public places.

**Success Criteria:**
- Achieve a high accuracy (e.g., >90%) on the test dataset.
- The model should be able to generalize well to new, unseen images.

#### 2. Data Understanding

**Data Collection:**
- The dataset consists of images of people with and without masks.
- The dataset is already divided into training, validation, and test sets.

**Data Description:**
- Images are in JPEG format.
- Each image is labeled as either "Mask" or "No Mask".

#### 3. Data Preparation

**Data Preprocessing:**
- Resize images to 224x224 pixels to match the input size of the VGG16 model.
- Normalize pixel values to the range [0, 1].

**Data Augmentation:**
- Apply data augmentation techniques such as rotation, zoom, and horizontal flip to increase the diversity of the training data.

**Data Generators:**
- Use `ImageDataGenerator` to create training, validation, and test data generators.

#### 4. Modeling

**Model Selection:**
- Use a pretrained VGG16 model for transfer learning.
- Add custom layers on top of the VGG16 base model for binary classification.

**Model Training:**
- Compile the model with an appropriate optimizer (e.g., Adam) and loss function (e.g., binary crossentropy).
- Train the model using the training data and validate it using the validation data.

**Hyperparameter Tuning:**
- Experiment with different learning rates, batch sizes, and number of epochs to find the best hyperparameters.

#### 5. Evaluation

**Model Evaluation:**
- Evaluate the model on the test dataset to measure its performance.
- Use metrics such as accuracy, precision, recall, and F1-score.

**Visual Inspection:**
- Display a few test images along with their predicted and actual labels to visually inspect the model's performance.

#### 6. Deployment

**Model Saving:**
- Save the trained model to a file (e.g., `mask_detection_vgg16.h5`).

**Model Deployment:**
- Deploy the model to a production environment where it can be used to make predictions on new images.
- Provide an interface (e.g., a web application) for users to upload images and get predictions.

**Monitoring and Maintenance:**
- Monitor the model's performance in the production environment.
- Retrain the model periodically with new data to maintain its accuracy.

### Files

- `mask_detector.ipynb`: Jupyter Notebook for developing the mask detection model.
- `requirements.txt`: List of required Python packages.

### Instructions

1. **Open the Jupyter Notebook**:
    ```sh
    jupyter notebook mask_detector.ipynb
    ```

2. **Run the Notebook**:
    Follow the instructions in the notebook to load the dataset, preprocess the data, build and train the model, and evaluate its performance.

3. **Save and Load the Model**:
    Save the trained model to a file and load it for making predictions on new images.

## HW6-2: Image Generation with Stable Diffusion

This assignment involves generating images based on text prompts using the Stable Diffusion model from the `diffusers` library.

### Files

- `img_generator.ipynb`: Jupyter Notebook for generating images using the Stable Diffusion model.
- `requirements.txt`: List of required Python packages.

### Instructions

1. **Open the Jupyter Notebook**:
    ```sh
    jupyter notebook img_generator.ipynb
    ```

2. **Run the Notebook**:
    Follow the instructions in the notebook to load the Stable Diffusion model, set the text prompt, and generate images.

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/YANGCHIHUNG/IoT-data-analysis-and-application
   cd HW6/HW6-2
   ```

2. **Create a virtual environment:**

   ```sh
   pip install -r requirements.txt
   ```