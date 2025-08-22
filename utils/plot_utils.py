import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

from torchvision import transforms
import cv2

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


def get_cam(img_path, model, plot=True):
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
    
    model.eval()
    
    target_layer = model.model.layer4[1].conv2
    
    # Temporarily set requires_grad to True for the target layer and its immediate parent.
    original_requires_grad = {}
    for name, param in model.named_parameters():
        if 'layer4' in name:
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

    # TO PASS AS ARG!
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
    ])
    
    original_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).requires_grad_(True)
    
    gradients = {}
    activations = {}

    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0].detach()
        return hook

    handle_activation = target_layer.register_forward_hook(save_activation("conv"))
    handle_gradient = target_layer.register_backward_hook(save_gradient("conv"))

    output = model.model(input_tensor)
    
    class_idx = output.argmax().item()
    
    score = output[0, class_idx]
    score.backward()

    grad = gradients["conv"][0]
    act = activations["conv"][0]

    # Compute weights and CAM.
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(0)
    cam = torch.relu(cam)

    # Normalize.
    cam -= cam.min()
    cam /= cam.max()

    # Convert CAM to numpy and resize.
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (original_img.size[0], original_img.size[1]))

    # Normalize again after resize.
    cam -= np.min(cam)
    cam /= np.max(cam)

    # Remove hooks.
    handle_activation.remove()
    handle_gradient.remove()

    # Restore the original requires_grad state.
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = original_requires_grad[name]
    
    # Overlay on original image and show.
    if plot:
        plt.figure()
        plt.imshow(original_img)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.title(f"Predicted - {class_names[class_idx]}")
        plt.axis("off")
        plt.show()

    return cam, class_idx