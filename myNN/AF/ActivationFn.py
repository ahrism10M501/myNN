import numpy as np
import abc

class ActivationFn(abc.ABC):
    @abc.abstractmethod
    def forward(self, inputs):
        pass
    
    @abc.abstractmethod
    def backward(self, dout):
        pass
    
    def __call__(self, inputs):
        return self.forward(inputs)