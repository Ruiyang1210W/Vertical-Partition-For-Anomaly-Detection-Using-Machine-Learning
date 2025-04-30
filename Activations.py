import numpy as np 
# Activation functions introduce non-linearity, allowing the network to solve more complex problems than it could with
# linear transformations alone.
# Sigmoid Acitvation (Common for binary classification)
# Formula: 𝜎(x) = 1/(1 + e^(-x))
# Derivative: 𝜎'(x) = 𝜎(x)*(1-𝜎(x))


class Activation:
    def activate(self, input):
        input = np.clip(input, -500, 500)  # Prevent overflow
        # Base activation function (Sigmoid)
        return 1/(1 + np.exp(-input))

    def derivative(self, input):
        # Derivative of sigmoid for backprop
        sig = self.activate(input)
        return sig * (1 - sig)


class ReLU:
    # ReLU is common in hidden layers due to its simplicity and efficiency
    # Formula: ReLU(x) = max(0,x)
    # Derivative: ReLU'(x) = 1 if x > 0 else 0
    def activate(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)
