import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

# Load the Auto MPG dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

# Load and preprocess the dataset
auto_mpg = pd.read_csv(url, names=column_names, delim_whitespace=True)

# Data cleaning - Replace missing '?' values with NaN and drop rows with missing values
auto_mpg.replace('?', np.nan, inplace=True)
auto_mpg.dropna(inplace=True)
auto_mpg['horsepower'] = auto_mpg['horsepower'].astype(float)

# Drop the 'car name' column, not relevant for prediction
auto_mpg.drop(['car name'], axis=1, inplace=True)

# Convert 'origin' to categorical using one-hot encoding
auto_mpg = pd.get_dummies(auto_mpg, columns=['origin'], prefix='origin')

# Define features (X) and target (y)
X = auto_mpg.drop('mpg', axis=1)
y = auto_mpg['mpg']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page with input form
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    cylinders = int(request.form['cylinders'])
    displacement = float(request.form['displacement'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['acceleration'])
    model_year = int(request.form['model_year'])
    origin = int(request.form['origin'])  # Can be 1, 2, or 3

    # Prepare the input data
    input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year]])

    # One-hot encode 'origin'
    if origin == 1:
        input_data = np.append(input_data, [1, 0, 0])
    elif origin == 2:
        input_data = np.append(input_data, [0, 1, 0])
    else:
        input_data = np.append(input_data, [0, 0, 1])

    # Scale the input data
    input_data = scaler.transform([input_data])

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Return the result on the web page with the input values retained
    return render_template('index.html', prediction=prediction, 
                           cylinders=cylinders, 
                           displacement=displacement, 
                           horsepower=horsepower, 
                           weight=weight, 
                           acceleration=acceleration, 
                           model_year=model_year, 
                           origin=origin)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
