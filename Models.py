# The model class ties the layers together to form a full neural network.
# It handles initialization, forward propagation, and connecting layers.
# Models.py
from Layers import Layer, InputLayer
import numpy as np
from Activations import ReLU, Activation
from sharedParameter import generate_shared_parameters

width = 1  # one neuron per layer
depth = 4   # 4 layers: Input, Hidden1, Hidden2, Output

# Generate shared parameters (weights and biases)
startweights, startbiases = generate_shared_parameters(width, depth, seed=55)

class Model:
    # Connect each layer to the next by setting weights and biases
    def __init__(self, width, depth, startweights=None, startbiases=None, GPU=False):
        self.layers = [InputLayer(width)]
        
        for i in range(depth - 2):
            # Use provided bias if available; otherwise default to zero.
            bias = startbiases[i] if startbiases is not None else [0] * width
            self.layers.append(Layer(width, bias, ReLU(), i + 1, GPU))
        
        # Final layer:
        bias = startbiases[depth - 2] if startbiases is not None else [0] * width
        self.layers.append(Layer(width, bias, Activation(), depth - 1, GPU))

        for i, layer in enumerate(self.layers):
            if i > 0:
                weights = startweights[i-1].T if startweights is not None else np.random.randn(width, width)
                layer.connectPreviousLayer(self.layers[i-1], weights)

    def forward(self, values):
        # Forward pass: set input and propagate through layers.
        self.layers[0].setOutput(values)
        for layer in self.layers[1:]:
            layer.forward()
        
    def backward(self, ytrue, learning_rate=0.01):
        error = self.layers[-1].output - ytrue
        for layer in reversed(self.layers[1:]):
            error = layer.backward(error, learning_rate)

    def update(self, learning_rate=0.01):
        for layer in self.layers[1:]:
            layer.updateWeights(learning_rate)

    def out(self):
        return self.layers[-1].output


# Instantiate the model with the shared parameters.
model = Model(width, depth, startweights, startbiases)

