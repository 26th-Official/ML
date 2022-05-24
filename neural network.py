from matplotlib.axis import Axis
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

X,y = spiral_data(100,3)

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

class Loss:
    def calculate(self,output,y):
        # here output is the negative likelihood from Categoricalentropy class
        sample_losses= self.forward(output,y)
        # print("Samples ",sample_losses[:5])
        data_loss = np.mean(sample_losses)
        # print("Y ",y[:5])
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        # Here y_pred is nothing but the final output from softmax activation
        # Here y_true is nothing but y form generated dataset
        # print("y_pred ",y_pred[:5])
        # print("y_true ",y_true[:5])
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1 - 1e-7)
        # We are clipping the value from Y_pred to prevent 0 appearing before we do log so that infinity dosent occur
        # print("y_pred_clipped ",y_pred_clipped[:5])
        # print("y_true-shape ",y_true.shape)
        # print("len_y_true-shape ",len(y_true.shape))

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        
        # print("correct_confidence ",correct_confidence[:5])
        negatice_log_likelihoods = -np.log(correct_confidence)
        # print("Negative_likelihood ",negatice_log_likelihoods[:5])
        return negatice_log_likelihoods

layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print("Layer output ",activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output,y)

print("loss: ",loss)




