
# CNN Implementation with PyTorch on CIFAR-10 Dataset

This project is an implementation of Convolutional Neural Networks (CNNs) using PyTorch on the CIFAR-10 dataset from torchvision. The goal is to train a CNN model that can accurately classify images of 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck).

The project consists of a Jupyter Notebook that guides you through the following steps:

0- Loading and Preprocessing the Data

1- Defining the CNN Model Architecture

2- Defining Optimizer and Loss

3- Training the Model

4- Evaluating the Model Performance


## Getting Started

To get started with this project, you'll need to have Python 3.x installed on your machine along with the following libraries:
```
PyTorch
Torchvision
NumPy
Matplotlib
```

You can install these libraries by running the following command:
```
pip install torch torchvision numpy matplotlib
```
Once you have these libraries installed, download the cnn-cifar10.ipynb notebook from this repository.

## Running the Notebook

To run the Jupyter Notebook, open your terminal and navigate to the directory where you saved the notebook. Then run the following command:

jupyter notebook cnn-cifar10.ipynb
This will open the notebook in your web browser. Follow the instructions in the notebook to load and preprocess the data, define the CNN model architecture, train the model, and evaluate its performance.

## Results

After training the CNN model on the CIFAR-10 dataset, you should see an accuracy of around 75-85% on the test set. You can experiment with different model architectures and hyperparameters to see if you can improve the performance.

## Acknowledgments

This project was inspired by the PyTorch scholarship challenge from Facebook AI. The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research (CIFAR).
