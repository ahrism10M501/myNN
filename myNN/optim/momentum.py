import numpy as np
from .Optimizer import Optimizer

class momentum(Optimizer):
		# SGD와 다르게 momentum 계수를 받아옵니다
    def __init__(self, params, lr, momentum):
        super().__init__(params, lr)
        self.momentum = momentum
        # 이전 속도들을 저장해둘 공간을 지정하고, 이를 초기화합니다
        self.velocities = {}
        self._init_vel()
    
    # 주어진 layer에 맞는 크기를 지정해 행렬을 만듭니다
    def _init_vel(self):
        for i, layer in enumerate(self.params):
            self.velocities[i] = {}
            if hasattr(layer, 'weights'):
		            # np.zeros_like 는 주어진 행렬과 같은 크기의 영행렬을 만듭니다
                self.velocities[i]['weights'] = np.zeros_like(layer.weights)
            if hasattr(layer, 'bias'):
                self.velocities[i]['bias'] = np.zeros_like(layer.weights)
        
    def step(self):
		    # 주어진 각 레이어에 대해 전부 실시합니다
        for i, layer in enumerate(self.params):
            if hasattr(layer, 'weights'):
		            # 먼저 현재 레이어에 알맞는 weights를 불러오고
                vel = self.velocities[i]['weights']
                # 우리가 아는 공식에 대입합니다
                # layer.dweights는 역전파 과정에서 구합니다
                vel = self.momentum * vel + self.lr * layer.dweights
                
                # 이 속도를 저장합니다
                self.velocities[i]['weights'] = vel
                
                # 계산한 변화값을 현재 가중치에서 빼줍니다
                layer.weights -= vel
                
            if hasattr(layer, 'bias'):
                vel = self.velocities[i]['bias']
                vel = self.momentum * vel + self.lr * layer.dbiases
                
                self.velocities[i]['bias'] = vel
                
                layer.bias -= vel

    def zero_grad(self):
		    # 각 레이어의 기울기를 초기화 해 다음 역전파 과정에 오류가 없도록 합니다
		    # 이때 속도는 계속 저장하며 사용합니다
        for layer in self.params:
            if hasattr(layer, 'dweights'):
                layer.dweights = np.zeros_like(layer.weights)
            if hasattr(layer, 'dbiases'):
                layer.dbiases = np.zeros_like(layer.bias)