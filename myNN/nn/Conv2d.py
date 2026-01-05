import numpy as np

class Conv2d:
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        self.incn = in_channel
        self.outcn = out_channel
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.bias = np.zeros((out_channel, 1, 1))
        
    def forward(self, x):
        # (batch, channel, height, width)
        n, c, h, w = x.shape
        
        # x를 패드해서 x에 넣으면 문제날 수 있음
        if self.padding > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_pad = x
        out_H = (h+ 2* self.padding- self.ksize ) // self.stride + 1
        out_W = (w+ 2* self.padding- self.ksize ) // self.stride + 1
        
        processed_x = np.zeros((n, self.outcn, out_H, out_W))
        
        for i in range(out_H):
            for j in range(out_W):
                h_st = i*self.stride
                w_st = j*self.stride
                portion_x = x_pad[:, :, h_st:h_st+self.ksize, w_st:w_st+self.ksize]
                
                for k in range(self.outcn):
                    processed_x[:, k, i, j] = np.sum(portion_x* self.weights[k, :, :, :], axis=(1, 2, 3)) + self.bias[k]
    
    # 진짜 backward는 자신이 없다
    def backward(self, dout, x_pad, learning_rate):
        """
        dout: 현재 층의 출력에 대한 기울기 (n, outcn, out_H, out_W)
        x_pad: forward에서 사용했던 패딩된 입력 (n, incn, H_pad, W_pad)
        """
        n, out_c, out_h, out_w = dout.shape
        _, in_c, k_h, k_w = self.weights.shape
        
        # 1. 초기화
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        dx_pad = np.zeros_like(x_pad)

        # 2. 기울기 계산 (Forward의 루프를 역으로 추적)
        for i in range(out_h):
            for j in range(out_w):
                h_st = i * self.stride
                w_st = j * self.stride
                
                # Forward에서 썼던 그 영역 (N, C_in, K, K)
                portion_x = x_pad[:, :, h_st:h_st+self.ksize, w_st:w_st+self.ksize]
                
                for k in range(out_c):
                    # db: 출력 오차의 합 (Batch와 공간 차원에 대해 합산)
                    db[k] += np.sum(dout[:, k, i, j])
                    
                    # dw: 입력 영역 * 출력 오차 (N 차원 합산)
                    # dout[:, k, i, j]는 (N,) 형태이므로 브로드캐스팅 활용
                    dw[k] += np.sum(portion_x * dout[:, k, i, j][:, None, None, None], axis=0)
                    
                    # dx: 가중치 * 출력 오차
                    dx_pad[:, :, h_st:h_st+self.ksize, w_st:w_st+self.ksize] += \
                        self.weights[k] * dout[:, k, i, j][:, None, None, None]

        # 3. 패딩 제거 (dx_pad에서 원래 x의 크기만 추출)
        if self.padding > 0:
            dx = dx_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_pad

        # 4. 가중치 업데이트 (Optimizer가 따로 없다면 여기서 직접 수행)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dx
    
    def __call__(self, x):
        return self.forward(x)