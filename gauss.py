import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
num_points = 10000
num_groups = 5
coordinate_range = (-300, 300)

# Define a color palette for the groups
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Gaussian function
def gaussian(x, m, sigma):
    return np.exp(-((x - m) ** 2) / (2 * sigma ** 2))

# Generate points
points = []
group_colors = []
digis=10000
for group in range(num_groups):
    mx = random.uniform(coordinate_range[0], coordinate_range[1])
    my = random.uniform(coordinate_range[0], coordinate_range[1])
    sigma_x = 10
    sigma_y = 10

    for _ in range(num_points // num_groups):
        while True:
            x = random.uniform(coordinate_range[0], coordinate_range[1])
            prob_x = gaussian(x, mx, sigma_x)
            #rand_prob = random.uniform(0, 1)
            rand_prob = int(random.uniform(0, 1)*digis)
            rand_prob = rand_prob/digis
            if prob_x > rand_prob:
                break

        while True:
            y = random.uniform(coordinate_range[0], coordinate_range[1])
            prob_y = gaussian(y, my, sigma_y)
            #rand_prob = random.uniform(0, 1)
            rand_prob = int(random.uniform(0, 1)*digis)
            rand_prob = rand_prob/digis
            if prob_y > rand_prob:
                break

        points.append((round(x, 2),round(y, 2)))
        group_colors.append(colors[group])  # Assign color based on group

# Save points to file
with open('generated_points.txt', 'w') as f:
    for point in points:
        f.write(f"{point[0]}, {point[1]}\n")

# Plot points with different colors for each group
plt.figure(figsize=(10, 10))
for group in range(num_groups):
    group_points = [(points[i][0], points[i][1]) for i in range(len(points)) if group_colors[i] == colors[group]]
    x_vals, y_vals = zip(*group_points)
    plt.scatter(x_vals, y_vals, color=colors[group], alpha=0.5, label=f'Group {group + 1}')

plt.xlim(coordinate_range)
plt.ylim(coordinate_range)
plt.title('Generated Points with Different Colors for Each Group')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid()
plt.legend()
plt.show()