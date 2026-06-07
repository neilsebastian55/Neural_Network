import numpy as np

class Layer:
    def __init__(self, inputs, neurons, activation):
        self.weights = np.random.randn(inputs, neurons) * 0.1
        self.bias = np.zeros(1, neurons)
        self.activation = activation
    
    def forward(self,x):
        self.input = x
        self.weighted_sum = x @ self.weights + self.bias
        self.output = self.activation.forward(self.weighted_sum)
        
