import numpy as np
from src.func.LossFn import LossFn

class BCELoss(LossFn):
    def forward(self, inputs, labels):
        epsilon = 1e-12
        inputs = np.clip(inputs, epsilon, 1.0 - epsilon) # 만약 0을 log에 넣으면 오류난다
        
        term1 = labels* np.log(inputs)
        term2 = (1-labels)* np.log(1-inputs)
        
        loss_total = -(term1 + term2)
        return loss_total.flatten()