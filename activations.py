# activations.py
"""
This file does not store knowledge.
It has no weights.
It does not learn.

It only defines how space is deformed after each projection.


This module defines the activation functions used in the neural network.

Activation functions introduce non-linearity into the system.
Without them, a neural network would be equivalent to a single linear layer.

They are responsible for:
- shaping the geometry of the space
- enabling complex decision boundaries
- making learning possible
"""

import numpy as np


def relu(x):
    """
    Rectified Linear Unit (ReLU)

    Applies the function:
        f(x) = max(0, x)

    Effect:
    - Removes negative values
    - Keeps positive values unchanged
    - Introduces sparsity and non-linearity

    Used in:
    - All hidden layers (standard practice)
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid activation function

    Applies the function:
        f(x) = 1 / (1 + e^(-x))

    Effect:
    - Maps any real value into the range (0, 1)
    - Interpretable as probability

    Used in:
    - Binary classification output layer
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax activation function

    Applies the function:
        f(x_i) = exp(x_i) / sum(exp(x_j))

    Effect:
    - Converts a vector into a probability distribution
    - All outputs are positive and sum to 1
    - Forces competition between classes

    Used in:
    - Multiclass classification output layer
    """

    # Numerical stability trick
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)