import numpy as np

class Linear:
    def __init__(self, input_size, output_size, bias=True):
        # Xavier initialization for better training stability
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.dweights = np.zeros((input_size, output_size))
        
        if bias:
            self.bias = np.zeros((1, output_size))
            self.dbiases = np.zeros((1, output_size))
        else:
            self.bias = 0
            self.dbiases = 0
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def backward(self, dout):
        """
        Backward pass for Linear layer
        
        Args:
            dout: Gradient of loss with respect to output (batch_size, output_size)
        
        Returns:
            dinput: Gradient of loss with respect to input (batch_size, input_size)
        """
        # Compute gradient with respect to weights
        self.dweights = self.inputs.T @ dout
        
        # Compute gradient with respect to bias
        if isinstance(self.bias, np.ndarray):
            self.dbiases = np.sum(dout, axis=0, keepdims=True)
        else:
            self.dbiases = 0
        
        # Compute gradient with respect to input (to pass to previous layer)
        dinput = dout @ self.weights.T
        
        return dinput
    
    def __call__(self, inputs):
        return self.forward(inputs)