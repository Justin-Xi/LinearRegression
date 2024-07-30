import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56])
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89])

# Reshape the data( from horizontal to vertical)
time_studied = time_studied.reshape(-1, 1)
scores = scores.reshape(-1, 1)

time_trained, time_tested, scores_trained, scores_tested = train_test_split(time_studied, scores, test_size=0.2)

# Initialize the model
model = LinearRegression()
model.fit(time_trained, scores_trained)

#Test the model
accuracy = model.score(time_tested, scores_tested)
print(f"Accuracy: {accuracy}")

# Visualize
plt.scatter(time_trained, scores_trained, color='blue')
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), color='red')
plt.ylim(0, 100)
plt.show()


