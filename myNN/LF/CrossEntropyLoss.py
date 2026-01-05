import numpy as np
from .LossFn import LossFn

class CrossEntropyLoss(LossFn):
    def forward(self, inputs, labels):
        samples = len(inputs)
        max_logit = np.max(inputs, axis=1, keepdims=True)
        
        log_sum_exp = max_logit + np.log(np.sum(np.exp(inputs - max_logit), axis=1, keepdims=True))
        log_softmax = inputs - log_sum_exp
        
        if len(labels.shape) == 1:
            correct_logit = log_softmax[range(samples), labels]
        elif len(labels.shape) == 2:
            correct_logit = np.sum(log_softmax*labels, axis=1)
        else:
            raise
            
        negative_log_likelihoods = -correct_logit
        return negative_log_likelihoods.flatten()
    
    def backward(self, dout, labels):
        samples, classes = dout.shape[0], dout.shape[1]
        
        if len(labels.shape) == 1:
            one_hot_labels = np.zeros((samples, classes))
            one_hot_labels[np.arange(samples), labels] = 1
            labels = one_hot_labels
        
        dx = -labels / dout
        return dx