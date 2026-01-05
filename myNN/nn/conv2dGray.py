import numpy as np

def conv2dGray(image, kernel, stride=1, padding=0):
    image = np.array(image)
    kernel = np.array(kernel)
    
    H, W = image.shape
    KH, KW = kernel.shape
    
    out_H = (H - KH + 2*padding)//stride + 1
    out_W = (W - KW + 2*padding)//stride + 1
    
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        
    processed_img = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            processed_img[i, j] = np.sum(image[i:i+KW*stride, j:j+KH*stride] * kernel)
    
    return processed_img