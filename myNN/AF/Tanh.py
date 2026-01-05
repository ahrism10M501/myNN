import numpy as np
from func.activationFn import ActivationFn

class Tanh(ActivationFn):
    def forward(self, inputs):
        self.outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))