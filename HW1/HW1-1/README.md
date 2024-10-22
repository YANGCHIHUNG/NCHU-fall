# HW1-1

![web](/HW1/HW1-1/images/web.png)

## Prompt

task:

write python to solve linear regression problem, flowing CRISP-DM steps 

action: 

1. generate n random points (x, y), x: -10 to 10, y = a*x + b + c*N(0, variance), where a is from -10 to 10, b = 50, c is from 0 to 100
2. draw regression line in red
3. put it on the flask web, and allow user to modify a in a*x + b, c, N

## CRISP-DM

### 1. Business Understanding
The goal is to build a web application where users can input variables (`a`, `b`, `N`) and visualize a dynamically generated linear regression plot based on these inputs.

* Data Mining Task:
Enable users to interact with and modify the coefficients of a linear regression model (e.g., slope `a`, intercept `b`, and noise `N`) and visually inspect the impact on the plot in real-time.

### 2. Data Understanding

- **Input Data**: The user provides the parameters `a`, `b`, and `N`, which represent:
  - `a`: The slope of the linear function.
  - `b`: The intercept of the linear function.
  - `N`: The standard deviation of the noise to add to the data.
  
- **Generated Data**: The code generates random points `x` and `y`:
  - `x`: Uniform random variable.
  - `y`: Generated using the linear equation `y = a * x + b` with Gaussian noise added.


### 3. Data Preparation

In this phase, we simulate data and prepare it for modeling.

### Data Generation:
```python
x = np.random.uniform(-10, 10, 100)
noise = np.random.normal(0, N, x.shape)
y = a * x + b + noise
```
We generate a random dataset of 100 points simulating noisy linear data based on user-provided parameters. No additional cleaning is needed because the data is synthetically generated.

### Features:
- `x`: Independent variable
- `y`: Dependent variable influenced by the model and noise


### 4. Modeling

We use linear regression to model the relationship between `x` and `y`.

### Model Selection:
We use **numpy's polyfit** function to fit a linear regression line:
```python
coeffs = np.polyfit(x, y, 1)
y_fit = coeffs[0] * x + coeffs[1]
```
This fits a straight line to the data points `x` and `y`, where `coeffs[0]` is the slope and `coeffs[1]` is the intercept of the fitted line.

### Visualization:
The model results are visualized using matplotlib:
```python
plt.scatter(x, y, label='Data points')
plt.plot(x, y_fit, color='red', label='Regression line')
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
```

### 5. Evaluation

The model is evaluated by visual inspection.

### Evaluation via Visualization:
The scatter plot of the data points along with the fitted regression line allows users to visually inspect the quality of the fit.

### User Feedback:
The web app allows users to modify the parameters and observe the updated results in real-time.


### 6. Deployment

The final phase involves deploying the model in a Flask web application.

### Flask Application Code:
The following code integrates the model into a Flask app for deployment.

```python
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    data = request.get_json()
    a = data['a']
    b = data['b']
    N = data['N']
    
    x = np.random.uniform(-10, 10, 100)
    noise = np.random.normal(0, N, x.shape)
    y = a * x + b + noise
    
    coeffs = np.polyfit(x, y, 1)
    y_fit = coeffs[0] * x + coeffs[1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points')
    plt.plot(x, y_fit, color='red', label='Regression line')
    plt.title('Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
```
The model is deployed in a web-based application, where users can input values for the linear model's parameters and receive a dynamic plot in return.

## How to Run the Application

### Prerequisites
* Python 3.x
* Flask
* NumPy
* Matplotlib

### Installation

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

### Usage
* The application will open a webpage where users can input values for a (slope), b (intercept), and N (noise).
* Click "Submit" to update the graph. The graph will display the data points and the regression line based on the input values.