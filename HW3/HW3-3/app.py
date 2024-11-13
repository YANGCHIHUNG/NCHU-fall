import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import streamlit as st

# Title for the Streamlit app
st.title("3D SVM Visualization with Gaussian Data")

# Step 1: Generate random points
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from the origin
distances = np.sqrt(x1**2 + x2**2)

# Assign labels Y=0 for points within distance 4, Y=1 for the rest
Y = np.where(distances < 4, 0, 1)

# Step 2: Calculate x3 as a Gaussian function of x1 and x2
def gaussian_function(x1, x2):
    return np.exp(-(x1**2 + x2**2) / (2 * variance))

x3 = gaussian_function(x1, x2)

# Step 3: Train a Linear SVC
X = np.column_stack((x1, x2, x3))
clf = LinearSVC()
clf.fit(X, Y)

# Step 4: Create a meshgrid for plotting decision boundary
xx, yy = np.meshgrid(np.linspace(x1.min(), x1.max(), 50), np.linspace(x2.min(), x2.max(), 50))

# Step 5: Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x1, x2, x3, c=Y, cmap='bwr', marker='o')

# Add a slider to adjust the height of the plane
plane_height = st.slider('Adjust plane height', float(x3.min()), float(x3.max()), float(x3.mean()))

# Plot the decision boundary plane
zz = np.full_like(xx, plane_height)  # Create a constant plane at the height of plane_height
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Show the plot in Streamlit
st.pyplot(fig)