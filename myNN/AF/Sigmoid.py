import numpy as np
from func.activationFn import ActivationFn

class Sigmoid(ActivationFn):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))