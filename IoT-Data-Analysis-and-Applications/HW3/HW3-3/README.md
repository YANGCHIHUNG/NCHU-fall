# 3D SVM Visualization with Gaussian Data

![web](/HW3/HW3-3/images/screenshot.png)

This is a Streamlit application that visualizes 3D data generated using a Gaussian function and classified using a Linear Support Vector Machine (SVM).

## Features

- Generate random Gaussian-distributed data points
- Classify data points using a Linear SVM
- Visualize data points and classification results in 3D
- Adjust the height of the decision boundary plane using a slider

## Installation

1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/YANGCHIHUNG/IoT-data-analysis-and-application.git
   cd HW3
   cd HW3-3
   ```
2. Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```


## Usage

1. Run the application:
   ```sh
   streamlit run app.py
   ```
2. Open your web browser and go to **http://localhost:8501** to view the application.

## Code Explanation

### Main Features

- Generate Random Points: Generate 600 random points using a Gaussian distribution and calculate their distances from the origin.
- Assign Labels: Label points as Y=0 if they are within a certain distance from the origin, otherwise label them as Y=1.
- Calculate x3: Define a Gaussian function gaussian_function to calculate the z-values (x3) based on x1 and x2.
- Train Linear SVM: Train a Linear Support Vector Machine using the generated data.
- Create Meshgrid: Create a meshgrid for plotting the decision boundary.
- Plot 3D Graph: Use Matplotlib to plot a 3D scatter plot of the data points and the decision boundary plane.
- Adjust Plane Height: Use a Streamlit slider to adjust the height of the decision boundary plane.
- Display Plot in Streamlit: Use st.pyplot to display the plot in the Streamlit application.

## Dependencies
- Flask
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
