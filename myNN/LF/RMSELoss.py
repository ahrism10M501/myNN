import numpy as np
from .LossFn import LossFn

class RMSELoss(LossFn):
    def forward(self, inputs, labels):
        mse_loss = np.square(inputs-labels).flatten()
        return np.sqrt(mse_loss).flatten()
    
    def backward(self, dout, labels):
        rmse = self.forward(dout, labels)
        if rmse == 0:
            return np.zeros_like(dout)
        return (dout-labels)/(rmse * dout.size)