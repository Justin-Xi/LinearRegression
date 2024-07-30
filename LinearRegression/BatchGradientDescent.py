import numpy as np
import matplotlib.pyplot as plt
# Dataset
X = np.array([1, 2, 3, 4])
Y = np.array([2, 3, 4, 5])

# Initialize parameters
m = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Number of data points
n = len(X)
cost_history = []
# Batch Gradient Descent
for _ in range(num_iterations):
    # Compute predictions
    Y_pred = m * X + b
    cost = (1 / n) * np.sum((Y - Y_pred) ** 2)
    cost_history.append(cost)

    # Compute the gradients
    dm = (-2 / n) * np.sum(X * (Y - Y_pred))
    db = (-2 / n) * np.sum(Y - Y_pred)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Results
print(f"Estimated parameters: m = {m}, b = {b}")

# Predictions
Y_pred = m * X + b
print(f"Predicted values: {Y_pred}")

# Plotting the cost function over iterations
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
