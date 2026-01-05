import numpy as np
from .ActivationFn import ActivationFn

class Softmax(ActivationFn):
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
        
    def backward(self, dout):
        pass