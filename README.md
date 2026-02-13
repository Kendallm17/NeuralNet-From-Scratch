# NeuralNet-From-Scratch
 Building a Neural Network Using Pure Linear Algebra

# NeuralNet-From-Scratch  
Building a Neural Network Using Pure Linear Algebra

Author: Kendall Chinchilla Araya  
Systems Engineer – AI Engineer in Training  
Costa Rica 🇨🇷  
2026

---

## Project Overview

This project is a complete implementation of a neural network built entirely from first principles, without using any machine learning frameworks.

The objective is not performance, but **understanding**.

Every component of modern deep learning is recreated manually using only:

- Vectors
- Matrices
- Linear transformations
- Bias
- Activation functions
- Forward propagation

No TensorFlow.  
No PyTorch.  
No Keras.  

Only mathematics.

---

## Core Idea

A neural network is not an intelligent system.  
It is a geometric system.

It transforms input vectors through a sequence of linear and non-linear transformations until the data becomes linearly separable in a higher-dimensional space.

In this project, we explicitly implement that process.

---

## Architecture
<img width="326" height="168" alt="image" src="https://github.com/user-attachments/assets/c27f3001-f8a7-4b67-be69-fa1b11333955" />

## Mathematical Foundation

This project is based entirely on applied linear algebra.

Each layer computes:

x_next = f(Wx + b)

Where:

- x = input vector
- W = weight matrix
- b = bias vector
- f = activation function

This formula is repeated for each layer.

The entire network is just a chain of geometric transformations.

## Technical Concepts Used

### 1. Vectors
All data is represented as vectors.

Each vector is a point in a high-dimensional space.

### 2. Matrices as Layers
Each layer is a matrix that transforms the space.

A layer does not store knowledge.  
It stores a geometric deformation.

### 3. Bias
Bias shifts the space.

It represents the initial tendency of the neuron.

### 4. Activation Functions
Activation functions introduce non-linearity.

They deform the space so complex patterns become separable.

Used activations:

- ReLU
- Sigmoid
- Softmax

### 5. Forward Propagation
The forward pass propagates information through the network.

No learning happens here.  
Only execution.

