"""
Main.py is the main entry point of the neural network system.
It is responsible for creating the network, providing an input vector, running the forward propagation process, and displaying the final output.

This file represents the moment where all mathematical and architectural components become an actual working model.

Responsibilities

This file performs four main tasks:

Defines the network architecture

Creates an example input vector

Executes the forward pass through the network

Prints the final output

"""
# main.py
import numpy as np
from network import NeuralNetwork

# Define network architecture
net = NeuralNetwork([3, 5, 4, 2], task="multiclass")

# Example input vector
x = np.array([0.5, 0.2, 0.9])

# Run forward pass
output = net.predict(x)

print("Network output:", output)