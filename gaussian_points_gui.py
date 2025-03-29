import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class GaussianPointsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaussian Points Generator")
        
        # Default centers
        self.centers = [
            {'mx': -200, 'my': -200, 'σx': 30, 'σy': 30},  # Bottom left
            {'mx': -200, 'my': 200, 'σx': 40, 'σy': 40},   # Top left
            {'mx': 200, 'my': 200, 'σx': 35, 'σy': 35},    # Top right
            {'mx': 200, 'my': -200, 'σx': 45, 'σy': 45},   # Bottom right
            {'mx': 0, 'my': 0, 'σx': 25, 'σy': 25}         # Center
        ]
        
        # Create main frames
        self.params_frame = ttk.LabelFrame(root, text="Center Parameters", padding="10")
        self.params_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        self.plot_frame = ttk.LabelFrame(root, text="Points Distribution", padding="10")
        self.plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")
        
        # Create parameter entry fields
        self.entries = []
        for i in range(5):
            center_frame = ttk.LabelFrame(self.params_frame, text=f"Center {i+1}", padding="5")
            center_frame.grid(row=i, column=0, padx=5, pady=5, sticky="nsew")
            
            center_entries = {}
            for j, param in enumerate(['mx', 'my', 'σx', 'σy']):
                ttk.Label(center_frame, text=f"{param}:").grid(row=0, column=j*2, padx=2)
                entry = ttk.Entry(center_frame, width=8)
                entry.insert(0, str(self.centers[i][param]))
                entry.grid(row=0, column=j*2+1, padx=2)
                center_entries[param] = entry
            self.entries.append(center_entries)
        
        # Create generate button
        self.generate_btn = ttk.Button(self.params_frame, text="Generate Points", command=self.generate_points)
        self.generate_btn.grid(row=5, column=0, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
    
    def gaussian_probability(self, x, m, σ):
        """Calculate Gaussian probability."""
        return np.exp(-((x - m) ** 2) / (2 * σ ** 2))
    
    def generate_coordinate(self, m, σ, domain=(-300, 300)):
        """Generate a coordinate using the specified algorithm."""
        while True:
            x = np.random.uniform(domain[0], domain[1])
            prob = self.gaussian_probability(x, m, σ)
            random_prob = np.random.random()
            if prob > random_prob:
                return x
    
    def generate_point(self, center):
        """Generate both x and y coordinates for a point."""
        x = self.generate_coordinate(center['mx'], center['σx'])
        y = self.generate_coordinate(center['my'], center['σy'])
        return x, y
    
    def update_centers_from_gui(self):
        """Update centers with values from GUI entries."""
        for i, entries in enumerate(self.entries):
            for param in ['mx', 'my', 'σx', 'σy']:
                try:
                    self.centers[i][param] = float(entries[param].get())
                except ValueError:
                    # If invalid input, keep default value
                    entries[param].delete(0, tk.END)
                    entries[param].insert(0, str(self.centers[i][param]))
    
    def generate_points(self):
        """Generate points and update plot."""
        self.update_centers_from_gui()
        
        # Generate points
        points = []
        for _ in range(10000):
            group = np.random.randint(0, len(self.centers))
            x, y = self.generate_point(self.centers[group])
            points.append((x, y, group))
        
        # Save points to file
        with open('points.txt', 'w') as f:
            for x, y, group in points:
                f.write(f"{x:.6f},{y:.6f},{group}\n")
        
        # Update plot
        self.ax.clear()
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Separate points by group
        groups = [[] for _ in range(len(self.centers))]
        for x, y, group in points:
            groups[group].append((x, y))
        
        # Plot each group
        for group_idx, group_points in enumerate(groups):
            if group_points:
                x, y = zip(*group_points)
                self.ax.scatter(x, y, c=colors[group_idx], label=f'Group {group_idx + 1}', alpha=0.6)
        
        # Configure plot
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title('Generated Points Distribution')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)
        self.ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # Save plot
        self.fig.savefig('points_distribution.png')
        
        # Update canvas
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = GaussianPointsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
