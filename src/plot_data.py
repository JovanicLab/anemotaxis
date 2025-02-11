import matplotlib.pyplot as plt
import numpy as np

def plot_larva_data(larva_data, larva_id, style_path=None):
    """Plots speed, length, and curvature over time for a single larva.
    
    Args:
        larva_data (dict): Dictionary containing time, speed, length, and curvature.
        larva_id (str): Identifier of the larva (for title).
        style_path (str, optional): Path to the .mplstyle file.
    """
    if style_path:
        plt.style.use(style_path)

    # Extract data
    time = np.array(larva_data["time"])
    speed = np.array(larva_data["speed"])
    length = np.array(larva_data["length"])
    curvature = np.array(larva_data["curvature"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(time, speed, label="Speed", color="blue", linewidth=2)
    ax.plot(time, length, label="Length", color="green", linewidth=2)
    ax.plot(time, curvature, label="Curvature", color="red", linewidth=2)

    # Labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Values")
    ax.set_title(f"Larva {larva_id}")

    # Remove grid
    ax.grid(False)

    # Set legend
    ax.legend()

    # Show the plot
    plt.show()
