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
    # Get parameters from the request
    data = request.get_json()
    a = data['a']
    b = data['b']
    N = data['N']
    
    # Generate random points
    x = np.random.uniform(-10, 10, 100)
    noise = np.random.normal(0, N, x.shape)
    y = a * x + b + noise
    
    # Create linear regression line
    coeffs = np.polyfit(x, y, 1)
    y_fit = coeffs[0] * x + coeffs[1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points')
    plt.plot(x, y_fit, color='red', label='Regression line')
    plt.title('Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    
    # Encode the image in base64
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
