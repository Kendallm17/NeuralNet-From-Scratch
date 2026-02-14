# layers.py
"""
This module defines the core structural unit of a neural network: the Layer.

A Layer contains only two things:
- A weight matrix (W)
- A bias vector (b)

This is the entire "brain" of the network.
All learning happens by modifying these two objects.

No activation functions.
No forward logic.
No training logic.

Only pure linear algebra.
"""

import numpy as np

class Layer:
    """
    Represents a fully connected (dense) neural network layer.
    
    Each layer performs the following mathematical operation:
        z = W @ x + b
    
    Where:
        - x is the input vector
        - W is the weight matrix
        - b is the bias vector
        - z is the linear output (pre-activation)
    """

    def __init__(self, input_size, output_size):
        """
        Initializes a layer with random weights and biases.

        Parameters:
        - input_size (int): number of input features
        - output_size (int): number of neurons in this layer
        """

        # Weight matrix: shape (output_size, input_size)
        self.W = np.random.randn(output_size, input_size)

        # Bias vector: shape (output_size,)
        self.b = np.random.randn(output_size)

    def __repr__(self):
        """
        Returns a readable representation of the layer.
        Useful for debugging and inspecting the network structure.
        """
        return f"Layer(W={self.W.shape}, b={self.b.shape})"