import numpy as np
from src.AF.ActivationFn import ActivationFn

class Softmax(ActivationFn):
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)