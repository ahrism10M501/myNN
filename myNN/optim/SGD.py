import numpy as np
from .Optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
    
    def step(self):
        for layer in self.params:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dweights
            if hasattr(layer, 'bias'):
                layer.bias -= self.lr * layer.dbiases
    
    def zero_grad(self):
        for layer in self.params:
            if hasattr(layer, 'dweights'):
                layer.dweights = np.zeros_like(layer.weights)
            if hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.bias)