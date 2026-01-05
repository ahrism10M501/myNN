import numpy as np

class Linear:
    def __init__(self, input_size, output_size, bias=True):
        self.weights = np.random.randn(input_size, output_size)
        if bias:
            self.bias = np.zeros((1, output_size))
        else:
            self.bias = 0
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def __call__(self, inputs):
        return self.forward(inputs)