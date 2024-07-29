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
num_epochs = 100

# Stochastic Gradient Descent
n = len(X)
cost_history = []
for epoch in range(num_epochs):
    total_cost = 0
    for i in range(n):
        # Compute the gradient for the current example
        x_i = X[i]
        y_i = Y[i]

        # Predictions and errors
        y_pred = m * x_i + b
        error = y_i - y_pred
        total_cost += error ** 2

        # Compute gradients
        dm = -2 * x_i * error
        db = -2 * error

        # Update parameters
        m = m - learning_rate * dm
        b = b - learning_rate * db
    cost_history.append(total_cost / n)

# Results
print(f"Estimated parameters: m = {m}, b = {b}")

# Predictions
Y_pred = m * X + b
print(f"Predicted values: {Y_pred}")

# Plotting the cost function over epochs
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
