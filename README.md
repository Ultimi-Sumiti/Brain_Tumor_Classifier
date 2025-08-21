## Brain Tumor Classifier

## Table of Contents
- [Brain Tumor Classifier](#brain-tumor-classifier)
- [Project's Goal](#projects-goal)
- [Dataset](#dataset)
- [Technologies and Tools](#technologies-and-tools)
- [Fine-Tuning Process](#fine-tuning-process)
- [Performance and Results](#performance-and-results)
  
This project addresses the problem of brain tumor classification from Magnetic Resonance Imaging (MRI) scans using deep learning techniques. The system is designed to provide accurate tumor detection from MRI data, highlighting the potential of deep learning in assisting medical diagnosis.


## Project's Goal
The primary goal of this project is to create a simple and efficient pipeline capable of fine-tuning a pre-trained convolutional neural network to solve a multi-class classification task. This system is designed to distinguish between healthy and pathological brain scans by including a "no tumor" class. The project aims to achieve good performance in classifying three types of brain tumors.

## Dataset
The dataset used in this work consists of 6000 MRI scans ([Paper](https://arxiv.org/abs/2506.14318%7D), [site](https://www.kaggle.com/datasets/briscdataset/brisc2025)). The images cover four classes: 
glioma, pituitary, meningioma, and no tumor. To ensure the trained model generalizes well, the images are taken from three different perspectives or planes: 
Coronal, Sagittal, and Axial.


## Technologies and Tools
The project was developed in Python using the PyTorch library. For the classification task, a Deep Convolutional Neural Network called [ResNet18](https://arxiv.org/abs/1512.03385%7D) was adopted as the backbone architecture due to its efficiency in image recognition. ResNet18 is a Residual Network with 18 layers and contains 11.2 million parameters.



Since the dataset is not large enough to train the entire model from scratch, a technique called fine-tuning was used.

# Fine-Tuning Process
The fine-tuning technique consists of the following main steps:

1. Copy the values of the pre-trained parameters into the model.

2. Rework the last one or two layers of the model, for example, by changing the number of output neurons to match the four classes being considered.

3. Freeze the values of all parameters except for the ones added in the previous step.

4. Train the new layers only.

5. Un-freeze the previous layers and fine-tune and continue training.

## Performance and Results
The experiments demonstrated strong classification performance with an accuracy that confirmed the model's reliability across different views. The results highlight the potential of deep learning frameworks in medical diagnosis. State-of-the-art applications of similar frameworks in computer vision have shown that their performance can be more reliable than that of many doctors from a statistical point of view.

Below are the results from the experiments, including the confusion matrix and evaluation metrics.
