import numpy as np
from .ActivationFn import ActivationFn

class ReLU(ActivationFn):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dout):
        dx = dout.copy()
        dx[self.inputs <= 0] = 0
        return dx