import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate random points
np.random.seed(42)  # For reproducibility
n = 100  # Number of points
a = np.random.uniform(-10, 10)  # Random slope between -10 and 10
b = 50  # Fixed intercept
c = np.random.uniform(0, 100)  # Random noise factor between 0 and 100
variance = 10  # Variance of the noise

x = np.random.uniform(-10, 10, n)  # Random x values between -10 and 10
noise = c * np.random.normal(0, variance, n)  # Noise
y = a * x + b + noise  # Linear relationship with noise

# Step 2: Fit a linear regression model
x_reshaped = x.reshape(-1, 1)  # Reshape x for sklearn
model = LinearRegression()
model.fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

# Step 3: Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Random Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression Example')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
