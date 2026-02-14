# network.py
"""
Build a neural network as a sequence of layers (matrices + bias) and 
provide a simple function to run forward flow over them.


Neural network structure definition.

This module defines the NeuralNetwork class, which represents
a fully connected feedforward neural network.

The network is defined as an ordered composition of layers.
It does not perform learning, backpropagation, or optimization.
It only defines the structure and provides an interface to
evaluate the network using forward propagation.

"""

from forward import forward
from layers import Layer


class NeuralNetwork:
    """
    Represents a fully connected feedforward neural network.

    This class stores an ordered list of layers and exposes
    a simple interface to run a forward pass.

    The network does not know how to learn.
    It only knows how to exist as a mathematical function.
    """
    def __init__(self, architecture, task="multiclass"):
        """
        Builds the neural network from a given architecture.

        Parameters
        ----------
        architecture : list of int
            Defines the size of each layer.

            Example:
                architecture = [3, 5, 4, 2]

            Means:
                Input layer: 3 neurons
                Hidden layer 1: 5 neurons
                Hidden layer 2: 4 neurons
                Output layer: 2 neurons
        """

        self.layers = []
        self.task = task

        for i in range(len(architecture) - 1):
            layer = Layer(
                input_size=architecture[i],
                output_size=architecture[i + 1]
            )
            self.layers.append(layer)

    def predict(self, x, task="multiclass"):
        """
        Evaluates the neural network on a given input.

        This performs a forward pass through all layers.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        task : str
            "binary", "multiclass", or "regression"

        Returns
        -------
        np.ndarray
            Output of the network.
        """

        return forward(self.layers, x, task)

    def __repr__(self):
        """
        Returns a human-readable representation of the network.
        """
        return f"NeuralNetwork({len(self.layers)} layers)"