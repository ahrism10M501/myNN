import abc
import numpy as np

class Optimizer(abc.ABC):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    @abc.abstractmethod
    def step(self):
        pass
    
    @abc.abstractmethod
    def zero_grad(self):
        pass