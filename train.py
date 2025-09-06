import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
#Defining the train, dev split function
def train_dev_split(data,label,train,dev):
    Ntr = int(data.shape[0] * train)
    Ndev = int(data.shape[0] * dev)
    ind = np.random.permutation(data.shape[0])
    data_shuffled = data[ind]
    label_shuffled = label[ind]
    Xtr = data_shuffled[:Ntr]
    Ytr = label_shuffled[:Ntr]
    Xdev = data_shuffled[Ntr:Ntr+Ndev]
    Ydev = label_shuffled[Ntr:Ntr+Ndev]
    return Xtr,Ytr,Xdev,Ydev
#Defining the relu function
def relu(arr):
    return np.maximum(np.zeros_like(arr), arr)
#Getting the data ready
print("Getting the data ready..")
data = idx2numpy.convert_from_file('data/train.idx3-ubyte')
labels = idx2numpy.convert_from_file('data/labels.idx1-ubyte')
B,W,H = data.shape[0], data.shape[1], data.shape[2]
data = data / data.reshape(B * W * H).max().item()
data = data.reshape(B, W* H)
Xtr,Ytr,Xdev,Ydev = train_dev_split(data,labels, 0.9,0.1)
#Initializing the weights
number_neurons_layer1 = 128
number_neurons_layer2 = 128
number_neurons_layer3 = 10
number_inputs = W*H
#Layer 1
W1 = np.random.randn(number_inputs, number_neurons_layer1) / np.sqrt(number_inputs) * np.sqrt(2)
b1 = np.random.randn(number_neurons_layer1) * 0
#Layer 2
W2 = np.random.randn(number_neurons_layer1, number_neurons_layer2) / np.sqrt(number_neurons_layer1) * np.sqrt(2)
b2 = np.random.randn(number_neurons_layer2) * 0
#Layer 3
W3 = np.random.randn(number_neurons_layer2, number_neurons_layer3) / np.sqrt(number_neurons_layer2) * np.sqrt(2)
b3 = np.random.randn(number_neurons_layer3) * 0
parameters = [W1,b1,W2,b2,W3,b3]

#TRAINING:
for i in range(2000):
    print("Training start..")
    print("NOTE: this might take several minutes.")
    #Mini-Batching
    batch_size = 64
    epsilon = 1e-9
    ind = np.random.randint(low=0,high=Xtr.shape[0],size=(batch_size,))
    mini_batch = Xtr[ind]
    correct_class_labels = Ytr[ind]

    #Forward pass
    layer1 = mini_batch @ W1 + b1
    tanhlayer = relu(layer1)
    layer2 = tanhlayer @ W2 + b2
    tanhlayer2 = relu(layer2)
    layer3 = tanhlayer2 @ W3 + b3
    softmax = np.exp(layer3) / np.sum(np.exp(layer3) + epsilon, axis = 1, keepdims=True)
    one_hot = np.zeros((batch_size,10))
    for k in range(batch_size):
        one_hot[k,Ytr[ind][k]] += 1
    cross_entropy_loss = 0
    correct_logprobs = -np.log(softmax[range(batch_size), correct_class_labels])
    cross_entropy_loss = np.mean(correct_logprobs)
    if i % 100 == 0:
        print(f'Loss on epoch {i} = {cross_entropy_loss}')

    #Backward pass
    dlayer3 = (softmax - one_hot) / batch_size
    dtanhlayer2 = dlayer3 @ W3.T #layer3 = tanhlayer2 @ W3 + b3
    dW3 = tanhlayer2.T @ dlayer3
    db3 = dlayer3.sum(0)  
    dlayer2 = dtanhlayer2 * (layer2 > 0) #tanhlayer2 = np.tanh(layer2)
    dtanhlayer = dlayer2 @ W2.T #layer2 = tanhlayer @ W2 + b2
    dW2 = tanhlayer.T @ dlayer2  
    db2 = dlayer2.sum(0)  
    dlayer1 = dtanhlayer * (layer1 > 0) #tanhlayer = np.tanh(layer1)
    dW1 = mini_batch.T @ dlayer1  #layer1 = mini_batch @ W1 + b1
    db1 = dlayer1.sum(0)
    gradient_params = [dW1,db1,dW2,db2,dW3,db3]
    lr = 5e-2 if i < 1000 else 1e-3
    for j in range(len(parameters)):
        parameters[j] = parameters[j] - lr * gradient_params[j]
    W1,b1,W2,b2,W3,b3 = tuple(parameters)
print("TRAINING COMPLETE")
#Evaluating on the dev dataset.
print("Now evaluating the model on the dev dataset...")
layer1 = Xdev @ W1 + b1
tanhlayer = relu(layer1)
layer2 = tanhlayer @ W2 + b2
tanhlayer2 = relu(layer2)
layer3 = tanhlayer2 @ W3 + b3
softmax = np.exp(layer3) / np.sum(np.exp(layer3) + epsilon, axis = 1, keepdims=True)
one_hot = np.zeros((Xdev.shape[0],10))
for k in range(Xdev.shape[0]):
    one_hot[k,Ydev] += 1
cross_entropy_loss = 0
correct_logprobs = -np.log(softmax[range(Xdev.shape[0]), Ydev])
cross_entropy_loss = np.mean(correct_logprobs)
print(f'Loss on the dev set = {cross_entropy_loss}')

print("Saving the model...")
np.savez("model_params.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

print("TRAINING AND SAVING COMPLETE")