import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for plots
import matplotlib.pyplot as plt
from flask import Flask, render_template

app = Flask(__name__)

# Step 1: Generate random data points
def generate_data(num_points=100):
    a = np.random.uniform(-10, 10)  # Random slope
    b = 50  # Fixed intercept
    c = np.random.uniform(0, 100, num_points)  # Random noise
    x = np.random.uniform(-10, 10, num_points)  # Random x values
    y = a * x + b + c  # Linear equation with noise
    return x, y

# Step 2: Create plot and save it to a file
def create_plot():
    x, y = generate_data()  # Generate data points

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data Points', color='blue', alpha=0.5)  # Scatter plot
    # Perform linear regression
    a, b = np.polyfit(x, y, 1)  # Fit a linear model
    plt.plot(x, a * x + b, label='Regression Line', color='red')  # Regression line
    plt.title('Linear Regression')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    
    # Save the plot to the static directory
    plt.savefig('static/plot.png')  
    plt.close()  # Close the plot

@app.route('/')
def index():
    create_plot()  # Create the plot when the index route is accessed
    return render_template('index.html', image_url='static/plot.png')

if __name__ == '__main__':
    # Create a static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
