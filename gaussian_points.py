import numpy as np
import matplotlib.pyplot as plt

# Define 5 centers with their parameters (mx, my, σx, σy)
CENTERS = [
    {'mx': -200, 'my': -200, 'σx': 30, 'σy': 30},  # Bottom left
    {'mx': -200, 'my': 200, 'σx': 40, 'σy': 40},   # Top left
    {'mx': 200, 'my': 200, 'σx': 35, 'σy': 35},    # Top right
    {'mx': 200, 'my': -200, 'σx': 45, 'σy': 45},   # Bottom right
    {'mx': 0, 'my': 0, 'σx': 25, 'σy': 25}         # Center
]

def gaussian_probability(x, m, σ):
    """Calculate Gaussian probability."""
    try:
        return np.exp(-((x - m) ** 2) / (2 * σ ** 2))
    except Exception as e:
        print(f"Error in gaussian_probability: {e}")
        print(f"Parameters: x={x}, m={m}, σ={σ}")
        raise

def generate_coordinate(m, σ, domain=(-300, 300)):
    """Generate a coordinate using the specified algorithm."""
    try:
        while True:
            # Step 2: Randomly choose a value
            x = np.random.uniform(domain[0], domain[1])
            
            # Step 3: Calculate probability using Gaussian function
            prob = gaussian_probability(x, m, σ)
            
            # Step 4: Generate random probability
            random_prob = np.random.random()
            
            # Step 5: Check if the point should be accepted
            if prob > random_prob:
                return x
    except Exception as e:
        print(f"Error in generate_coordinate: {e}")
        print(f"Parameters: m={m}, σ={σ}, domain={domain}")
        raise

def generate_point(center):
    """Generate both x and y coordinates for a point."""
    try:
        x = generate_coordinate(center['mx'], center['σx'])
        y = generate_coordinate(center['my'], center['σy'])
        return x, y
    except Exception as e:
        print(f"Error in generate_point: {e}")
        print(f"Center parameters: {center}")
        raise

def generate_dataset(n_points=10000):
    """Generate the complete dataset."""
    try:
        print("Starting to generate points...")
        points = []
        
        for i in range(n_points):
            if i % 1000 == 0:  # Print progress every 1000 points
                print(f"Generated {i} points...")
                
            # Step 1: Randomly select a group
            group = np.random.randint(0, len(CENTERS))
            x, y = generate_point(CENTERS[group])
            points.append((x, y, group))
        
        print(f"Successfully generated all {n_points} points")
        return points
    except Exception as e:
        print(f"Error in generate_dataset: {e}")
        raise

def save_points(points, filename='points.txt'):
    """Save points to a file."""
    try:
        print(f"Saving points to {filename}...")
        with open(filename, 'w') as f:
            for x, y, group in points:
                f.write(f"{x:.6f},{y:.6f},{group}\n")
        print(f"Successfully saved points to {filename}")
    except Exception as e:
        print(f"Error saving points to file: {e}")
        raise

def plot_points(points):
    """Visualize the generated points."""
    try:
        print("Creating visualization...")
        # Separate points by group
        groups = [[] for _ in range(len(CENTERS))]
        for x, y, group in points:
            groups[group].append((x, y))
        
        # Plot each group with different color
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        plt.figure(figsize=(10, 10))
        
        for group_idx, group_points in enumerate(groups):
            if group_points:
                x, y = zip(*group_points)
                plt.scatter(x, y, c=colors[group_idx], label=f'Group {group_idx + 1}', alpha=0.6)
        
        plt.grid(True)
        plt.legend()
        plt.title('Generated Points Distribution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Set axis limits to match the coordinate range
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)
        
        # Add coordinate system lines
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        print("Saving plot to points_distribution.png...")
        plt.savefig('points_distribution.png')
        plt.close()
        print("Successfully saved plot")
    except Exception as e:
        print(f"Error in plot_points: {e}")
        raise

def main():
    try:
        print("Starting program...")
        # Generate the dataset
        points = generate_dataset()
        
        # Save points to file
        save_points(points)
        
        # Create visualization
        plot_points(points)
        
        print("Done! Generated 10000 points across 5 centers.")
        print("Results saved in 'points.txt' and visualization in 'points_distribution.png'")
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
