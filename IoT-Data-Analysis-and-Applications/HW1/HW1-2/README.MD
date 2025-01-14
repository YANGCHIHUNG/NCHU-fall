# HW1-1

![web](/HW1/HW1-2/images/web.png)

## Prompt

Write Python code to predict the Auto MPG using Scikit-Learn by using linear regression, and following CRISP-DM, and show on the web by flask

## CRISP-DM

### 1. Business Understanding

Goal: Predict the miles per gallon (MPG) of vehicles based on various features.
Objective: Create a web application that allows users to input vehicle parameters and get MPG predictions.

### 2. Data Understanding

Load the Auto MPG dataset, understand its structure, and clean it.
```bash
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
```

### 3. Data Preparation

Data cleaning - Replace missing '?' values with NaN and drop rows with missing values

### 4. Modeling

Train a linear regression model
```bash
model = LinearRegression()
model.fit(X_train, y_train)
```

### 5. Evaluation

Evaluate the model
```bash
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
```

### 6. Deployment

Initialize the Flask app
```bash
app = Flask(__name__)
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YANGCHIHUNG/IoT-data-analysis-and-application.git
    cd HW1/HW1-1
    ```
2.  Install the required dependencies using requirements.txt:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python app.py
    ```
4. Open your browser and go to http://127.0.0.1:5000.

## How to Use the Application

1. Fill in the input parameters in the provided form:
* Cylinders: Number of cylinders in the vehicle.
* Displacement: Engine displacement in cubic inches.
* Horsepower: Engine horsepower.
* Weight: Vehicle weight in pounds.
* Acceleration: Time taken to accelerate from 0 to 60 mph in seconds.
* Model Year: Year of the vehicle model.
* Origin: Origin of the vehicle (1 for USA, 2 for Europe, 3 for Japan).
2. Click the "Predict" button to receive the estimated MPG.
3. The input values will remain visible after you press "Predict," and the predicted MPG will be displayed below the form.