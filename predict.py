import numpy as np
def make_prediction(input):
    #Takes in only one input
    def relu(arr):
        return np.maximum(np.zeros_like(arr), arr)
    file = np.load("model_params.npz")
    W1,b1,W2,b2,W3,b3 = file["W1"],file["b1"],file["W2"],file["b2"],file["W3"],file["b3"]
    W,H = input.shape[0], input.shape[1]
    input = input / input.reshape(W * H).max().item()
    input = input.reshape(W* H)
    epsilon = 1e-9
    layer1 = input @ W1 + b1
    tanhlayer = relu(layer1)
    layer2 = tanhlayer @ W2 + b2
    tanhlayer2 = relu(layer2)
    layer3 = tanhlayer2 @ W3 + b3
    softmax = np.exp(layer3) / np.sum(np.exp(layer3) + epsilon, keepdims=True)
    return softmax.argmax()
