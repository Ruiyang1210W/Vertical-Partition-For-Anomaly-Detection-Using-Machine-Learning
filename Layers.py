#Layer and Inputlayer classes
# Layers represent the neural network's structure
# Each layer should store its weights, biases, and activation function

# Input Layer, this layer holds the input value and passes them to the first hidden layer.
import numpy as np
from Activations import ReLU

class InputLayer:
    def __init__(self,width):
        self.output = np.zeros(width)

    def setOutput(self, values):
        self.output = np.array(values)

    def getNodes(self):
        return self.output

#Layer Class manages forward propagation, weights, biases, and activation for hidden/output layers.
class Layer:
    def __init__(self, width, biases, activation, layer_index, GPU=False):
        self.width = width
        self.biases = np.array(biases)
        self.activation = activation
        self.layer_index = layer_index
        self.output = np.zeros(width)
        self.weights = None
        self.prev_layer = None
        self.gradient = None  # Added for backward propagation

    def connectPreviousLayer(self, prev_layer, weights):
        #Connect to the previous layer with weights.
        self.weights = np.array(weights)
        self.prev_layer = prev_layer

    def forward(self):
       # z (pre-activation) is weighted sum of inputs plus bias for each neuron. z = input*weights + bias
        z = np.dot(self.prev_layer.output, self.weights) + self.biases
        # print(f"Layer {self.layer_index} pre-activation (z):", z)
        # output = activation(z)
        self.output = self.activation.activate(z)
        # print(f"Layer {self.layer_index} output:", self.output)
        return self.output
    
    def backward(self, error, learning_rate):
        activation_derivative = self.activation.derivative(self.output)
        self.gradient = error * activation_derivative
        weight_update = np.outer(self.prev_layer.output, self.gradient)

        self.weights -= learning_rate * weight_update
        self.biases -= learning_rate * self.gradient

        return np.dot(self.gradient, self.weights.T)  # Backpropagate error

    def updateWeights(self, learning_rate):
        self.weights -= learning_rate * self.gradient
    
     