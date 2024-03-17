import os
import re

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torch.utils.tensorboard

import keras.backend as K

# Assuming 'x' is your input tensor
x = K.placeholder(shape=(None,))

# Define the tanh function
tanh_output = K.tanh(x)

# Compute the derivative of tanh with respect to its inputs
derivative = K.gradients(tanh_output, x)

# Create a Keras function to evaluate the derivative
compute_derivative = K.function([x], [derivative])

# Test the derivative computation
input_value = np.array([0.5, 1.0, -0.5])
print(compute_derivative([input_value])[0])
