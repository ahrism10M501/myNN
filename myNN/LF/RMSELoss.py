import numpy as np
from myNN.LF.LossFn import LossFn

class RMSELoss(LossFn):
    def forward(self, inputs, labels):
        mse_loss = np.square(inputs-labels).flatten()
        return np.sqrt(mse_loss).flatten()
    
    def backward(self, inputs, labels):
        rmse = self.forward(inputs, labels)
        if rmse == 0:
            return np.zeros_like(inputs)
        return (inputs-labels)/(rmse * inputs.size)