import numpy as np
from .LossFn import LossFn

class MSELoss(LossFn):
    def forward(self, inputs, labels):
        return np.mean(np.square(inputs-labels).flatten())
    
    def backward(self, dout, labels):
        return 2*(labels-dout) / dout.size