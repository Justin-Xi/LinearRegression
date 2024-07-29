import numpy as np
import matplotlib.pyplot as plt


# Sample function: f(x) = x^2
def f(x):
    return x ** 2


# Gradient of the function: f'(x) = 2x
def gradient(x):
    return 2 * x


# Gradient descent parameters
learning_rate = 0.3
max_iterations = 100
tolerance = 1e-6

# Initial point
x = 10
history = [x]

# Gradient descent loop
for i in range(max_iterations):
    grad = gradient(x)
    new_x = x - learning_rate * grad
    history.append(new_x)

    # Check for overshooting
    if f(new_x) > f(x):
        print(f"Overshooting detected at iteration {i}")
        break

    # Check for convergence
    if abs(new_x - x) < tolerance:
        print(f"Converged at iteration {i}")
        break

    x = new_x

# Plotting the function and the path taken by gradient descent
x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label='f(x) = x^2')
plt.plot(history, f(np.array(history)), 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
