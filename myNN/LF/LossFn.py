import numpy as np
import abc

class LossFn(abc.ABC):
    @abc.abstractmethod
    def forward(self, inputs, labels) -> np.ndarray: pass
    
    @abc.abstractmethod
    def backward(self, dout, labels):
        if self.data_loss is None:
            raise ValueError("Loss must call calculate() before backward")
        pass
        
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        self.data_loss = np.mean(sample_losses)
        return self.data_loss