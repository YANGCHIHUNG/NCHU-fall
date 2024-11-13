# HW3-2

![web](/HW1/HW1-1/images/screenshot.png)

## Overview
This project demonstrates a 3D visualization of data using a Gaussian function and a Linear Support Vector Machine (SVM) for classification. The application is built using Streamlit for interactive visualization.

## Setup and Installation
1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd HW3
    cd HW3-2
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application
1. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

2. **Open your browser** and go to `http://localhost:8501` to view the application.

## Usage
- The application generates random data points and visualizes them in a 3D scatter plot.
- It trains a Linear SVM model to classify the data points.
- The separating hyperplane is visualized in the 3D plot.

## Dependencies
- Flask
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit