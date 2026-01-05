import numpy as np
from myNN.LF.LossFn import LossFn

class MSELoss(LossFn):
    def forward(self, inputs, labels):
        return np.mean(np.square(inputs-labels).flatten())
    
    def backward(self, inputs, labels):
        return 2*(labels-inputs) / inputs.size