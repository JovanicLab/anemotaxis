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

    # Create the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot speed
    ax1.plot(time, speed, label="Speed", color="blue", linewidth=2)
    ax1.set_ylabel("Speed")
    ax1.legend()
    ax1.grid(False)

    # Plot length
    ax2.plot(time, length, label="Length", color="green", linewidth=2)
    ax2.set_ylabel("Length")
    ax2.legend()
    ax2.grid(False)

    # Plot curvature
    ax3.plot(time, curvature, label="Curvature", color="red", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Curvature")
    ax3.legend()
    ax3.grid(False)

    # Set the title for the entire figure
    fig.suptitle(f"Larva {larva_id}")

    # Show the plot
    plt.show()