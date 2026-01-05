import numpy as np
from .ActivationFn import ActivationFn

class Sigmoid(ActivationFn):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dout):
        pass