import numpy as np
import nnfs
from nnfs.datasets import spiral_data

X,Y = spiral_data(100,3)

np.random.seed(0)
class Layer_Dense:
    def __init__ (self,n_input,n_neuron):
        self.weight = 0.1 * np.random.randn(n_input,n_neuron)
        self.bias = np.zeros((1,n_neuron))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weight) + self.bias

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

layer1 = Layer_Dense(2,4)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(4,3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])




