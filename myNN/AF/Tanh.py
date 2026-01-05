import numpy as np
from .ActivationFn import ActivationFn

class Tanh(ActivationFn):
    def forward(self, inputs):
        self.outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
        return self.outputs
    
    def backward(self, dout):
        pass