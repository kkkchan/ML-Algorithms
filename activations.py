import numpy as np 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    # 不要改变原数据
    x_ = np.copy(x)
    x_[x_ < 0] = 0
    return x_

def sigmoid_backward(dA, Z):
    return dA * sigmoid(Z) * (1 - sigmoid(Z))

def relu_backward(dA, Z):
    dA_ = np.copy(dA)
    dA_[Z<0] = 0
    return dA_ 