import matplotlib.pyplot as plt
import os

def plot_distribution(dataset_dir, name):
    class_names= ["glioma", "meningioma", "no_tumor", "pituitary"]
    
    frequencies = [
        num_files_in(os.path.join(dataset_dir, name.lower(), class_n))
        for class_n in class_names
    ]
    
    # Plot histogram.
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(class_names, frequencies, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
    
    # Add counts on top of each bar
    for bar, freq in zip(bars, frequencies):
        ax.text(
            bar.get_x() + bar.get_width()/2,  # x position (center of bar)
            bar.get_height(),                 # y position (top of bar)
            str(freq),                        # text (the count)
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    ax.set_title(f"Class Distribution in {name} Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_yticks([])
    plt.show()

def num_files_in(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])