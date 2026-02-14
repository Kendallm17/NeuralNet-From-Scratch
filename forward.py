# forward.py
"""
Forward propagation engine for a fully connected neural network.

This module implements the forward pass, which is the core computational
process of a neural network. It takes an input vector and propagates it
through a sequence of layers, applying linear transformations and
non-linear activation functions.

The forward pass does NOT perform learning.
It simply evaluates the mathematical function represented by the network.

Conceptually:
    Input → Projection → Deformation → Projection → ... → Output

The final activation function depends on the type of task:
    - Binary classification → Sigmoid
    - Multiclass classification → Softmax
    - Regression → Linear (no activation)
"""

import numpy as np
from activations import relu, sigmoid, softmax


def forward(layers, x, task="multiclass"):
    """
    Executes a forward pass through the neural network.

    This function evaluates the neural network as a pure mathematical
    function. It does not modify any parameters and does not perform
    optimization or learning.

    Each layer applies:
        1. A linear transformation: z = W @ a + b
        2. A non-linear deformation: a = activation(z)

    Parameters
    ----------
    layers : list of Layer
        Ordered list of Layer objects.
        Each Layer must contain:
            - W: weight matrix
            - b: bias vector

    x : np.ndarray
        Input vector.
        Shape must match the input size of the first layer.

    task : str, optional
        Specifies the type of problem being solved.
        Determines the activation of the final layer.

        Possible values:
            - "binary"     → Sigmoid output
            - "multiclass" → Softmax output
            - "regression" → Linear output

    Returns
    -------
    np.ndarray
        Output vector of the network.
        Interpretation depends on the task:
            - Binary → probability in [0,1]
            - Multiclass → probability distribution
            - Regression → continuous value(s)
    """

    # Current activation (starts as the raw input)
    a = x

    # Iterate through all layers
    for i, layer in enumerate(layers):

        # Linear transformation
        z = layer.W @ a + layer.b

        # Hidden layers: ReLU
        if i < len(layers) - 1:
            a = relu(z)

        # Output layer: task-specific activation
        else:
            if task == "binary":
                a = sigmoid(z)
            elif task == "multiclass":
                a = softmax(z)
            elif task == "regression":
                a = z  # identity function (no activation)

    return a