# From-Scratch MNIST Classifier (Numpy)

This project is a simple **3-layer neural network** trained to classify handwritten digits from the MNIST dataset.
Everything is implemented **from scratch in numpy**, no PyTorch, no TensorFlow.

This project consists of two python files and two jupyter notebooks. The model jupyter notebook is simply a notebook version of the train.py python file (which, as the name would suggest, just trains the model), just to make tinkering with the model easier.

The predict.py file contains one importable function that makes predictions from a single image, the test.ipynb file simply gives you a way to get the accuracy of the model on the test set.

## Features

- Fully connected MLP: 784 -> 100 -> 100 -> 10

-ReLU activatoins + Softmax output

-Cross-entropy loss with He initialization

-Manual forward and backward propagation

-Training & test evaluation (achieves roughly 95% accuracy)

-Model weights saved/loaded using 'numpy.savez'

-Simple predict module for single-digit inference

## Usage

# Training

``` python train.py ```

This saves the trained weights as model.npz

# Inference

```
from predict import make_prediction
import numpy as np
#Example: load a single MNIST image as a numpy array with shape (28,28)
image = ... #shape(28,28)
label = make_prediction(image)
print(f"Predicted digit: {label}")
```
## Results

-Training accuracy: 98%

-Test accuracy: 95%

-Cross entropy loss: as low as 0.1 on training data, 0.2 on dev data

## Why this project?

I built this to understand the mathematics of backpropagation and how neural networks actually learn under the hood.

It's a small project, but it covers the full pipeline: training -> saving -> loading -> inference

You're free to toy around with the hyperparameters as you wish and try to get as high of an accuracy as you can, enjoy!