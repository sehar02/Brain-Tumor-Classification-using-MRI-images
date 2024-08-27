# Brain Tumor Classification using MRI Images

This project aims to classify brain tumors using MRI images into different categories, such as glioma, meningioma, pituitary tumor, and nontumor, by leveraging deep learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Input and Output Data Format](#input-and-output-data-format)
- [Results](#results)

- [Conclusion](#conclusion)


## Project Overview
This project utilizes a Convolutional Neural Network (CNN) with attention layer to classify MRI brain images into different categories of tumors. The model was trained using TensorFlow and Keras, and the training history was plotted to observe the model's performance over time.

## Dataset
The dataset used for this project can be obtained from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It contains images labeled into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- Nontumor

## Installation
To run this project, you'll need to have Python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn
```

## Usage
Clone the repository:
bash
```bash

git clone https://github.com/Sehar02/Brain-Tumor-Classification-using-MRI-images.git
```
Navigate to the project directory:
bash
```bash
cd Brain-Tumor-Classification-using-MRI-images
```
Run the Jupyter Notebook: Open the Brain_Tumor_Classification.ipynb file in Jupyter Notebook and execute the cells to train and evaluate the model.
## Model Architecture
The model is a CNN-based architecture with the following layers:

Convolutional layers with ReLU activation
MaxPooling layers for downsampling
Dense layers for classification
Softmax activation for the output layer
Hyperparameters
Batch size: 32
Epochs: 25
Learning Rate: 0.001
Optimizer: Adam
## Training and Evaluation
The model was trained on the dataset using the following steps:

Data preprocessing, including resizing images to 224x224 pixels.
Splitting the dataset into training, validation, and test sets.
Training the model and monitoring the performance using validation data.
Input and Output Data Format
Input Data Format
Format: JPEG or PNG images.
Resolution: 224x224 pixels.
Color Mode: RGB.
## Output Data/Plots/Tables
Training Loss and Accuracy Plots: Graphs showing the model's loss and accuracy over each epoch for both training and validation sets.

Confusion Matrix: A matrix that summarizes the classification performance.

Classification Report: A report that includes precision, recall, and F1-score for each class.
## Results
The model achieved the following performance:

precesion: 98
recall: 98
Test Accuracy: 98


## Conclusion
This README provides a comprehensive guide for understanding and running the Brain Tumor Classification project. Follow the steps carefully to replicate the results or to build upon this work for further research.
