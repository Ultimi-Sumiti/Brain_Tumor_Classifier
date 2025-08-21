import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

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

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for (idx, path), name in zip(enumerate(rand_files), class_names):
        orig = Image.open(path)
        trans =transform(orig)

        ax_orig = axes[0, idx]
        ax_trans = axes[1, idx]

        ax_orig.imshow(orig, cmap='gray')
        ax_orig.set_title(f'{name} - Original', fontsize=20)
        ax_orig.axis('off')

        ax_trans.imshow(trans.permute(1, 2, 0).numpy(), cmap='gray')

        ax_trans.set_title(f'{name} - Transformed', fontsize=20)
        ax_trans.axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(brisc_dm, model):
    
    # Getting test dataloader.
    test_ds = brisc_dm.test_dataloader()
    
    # Setting the model in evaluation mode.
    model.eval()
    
    y_pred = []
    y_true = []
    
    with torch.no_grad(): 
        for images, labels in test_ds:
            # Computing the outputs.
            outputs = model(images)
            # For the same image get the class with higher probability.
            preds = torch.argmax(outputs, dim=1)
            # Adding labels and predictions.
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
    
    # Classification report.
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    
    # Confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix.
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    return cm

def plot_statistics(cm):
    
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
    
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)
    
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    metrics_df = pd.DataFrame({
    "Class": class_names,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score
    })

    metrics_df

    metrics_df.plot(x="Class", y=["Accuracy", "Precision", "Recall", "F1 Score"], kind="bar", figsize=(8, 5))
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
