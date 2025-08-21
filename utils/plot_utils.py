import matplotlib.pyplot as plt
from PIL import Image

import os
import random


def num_files_in(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


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


def plot_samples(dataset_dir, transform):
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

    class_dir = [os.path.join(dataset_dir, "train", name) for name in class_names]

    class_fn = [
        [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for directory in class_dir
    ]

    rand_files = [random.choice(rnd) for rnd in class_fn]
    print(rand_files)

    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    for (idx, path), name in zip(enumerate(rand_files), class_names):
        orig = Image.open(path)
        trans =transform(orig)

        ax_orig = axes[idx, 0]
        ax_trans = axes[idx, 1]

        ax_orig.imshow(orig)
        ax_orig.set_title(f'{name} - Original')
        ax_orig.axis('off')

        ax_trans.imshow(trans.permute(1, 2, 0).numpy())

        ax_trans.set_title(f'{name} - Transformed')
        ax_trans.axis('off')

    plt.tight_layout()
    plt.show()
