import numpy as np
from myNN.AF.ActivationFn import ActivationFn

class ReLU(ActivationFn):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, inputs):
        dx = inputs.copy()
        dx[self.inputs <= 0] = 0
        return dx