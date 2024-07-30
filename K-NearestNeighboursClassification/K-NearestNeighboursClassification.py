import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {'blue': [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
          'orange': [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]]}

new_point = [3, 3]


class K_Nearest_Neighbors:

    def __init__(self, k=3):
        self.k = k

    def fit(self, points):
        self.points = points

    def euclidean_distance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = self.euclidean_distance(point, new_point)
                distances.append([distance, category])

        # Get the categories of the k nearest points, the second actegory is just a variable name
        categories = [category[1] for category in sorted(distances)[:self.k]]
        # Counter(categories): Creates a Counter object from the list of categories. A Counter object is a dictionary
        # .most_common(1): Returns a list of the n most common elements and their counts from a Counter object. In
        # this case, n is 1, so it returns the most common category and its count as a list of one tuple.
        # [0]: Access the first element of the list returned by most_common(1), which is a tuple containing the most common category and its count.
        # [0]: Access the first element of the tuple, which is the most common category.
        result = Counter(categories).most_common(1)[0][0]
        return result


clf = K_Nearest_Neighbors(k=3)
clf.fit(points)
print(clf.predict(new_point))


# Plotting the data points
ax = plt.subplot()
ax.grid(True, color='#323232')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

[ax.scatter(point[0], point[1], color='#104DCA', s=60) for point in points['blue']]
[ax.scatter(point[0], point[1], color='#EF6C35', s=60) for point in points['orange']]


new_class = clf.predict(new_point)
color = '#EF6C35' if new_class == 'orange' else '#104DCA'
ax.scatter(new_point[0], new_point[1], color=color, marker='*', s=200, zorder=100)

[ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#104DCA', linestyle='--', linewidth=1) for point in points['blue']]
[ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#EF6C35', linestyle='--', linewidth=1) for point in points['orange']]

plt.show()