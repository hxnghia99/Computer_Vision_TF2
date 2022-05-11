import numpy as np


"""#######################
Page 30: SimpleModel_Numpy
#########################"""
class Neuron(object):
    """A simple feed-forward artificial neuron"""
    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        #Randomly initializing weights and bias
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn

    def forward(self, x):
        #Forward the input signal through the neuron
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)

#fixing the random generator seed
# np.random.seed(42)
#settings
num_x = 1000
#step activation function
fn = lambda x: 1 if x>0 else 0
#create neuron
neuron = Neuron(num_x, fn)
#create inputs
x = np.random.rand(num_x).reshape(1,num_x)
#feed-forward
out = neuron.forward(x)
# print(out)
# print(neuron.W, '\n', neuron.b)



"""#######################
Page 33: Fully Connected Layer
#########################"""
class FullyConnectedLayer(object):
    def __init__(self, num_inputs, layer_size, activation_fn):
        super().__init__()
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)
#initial settings
# np.random.seed(10)
num_inputs = 3
layer_size = 4
#create inputs
x1 = np.random.uniform(-1, 1, num_inputs).reshape(1, num_inputs)
x2 = np.random.uniform(-1, 1, num_inputs).reshape(1, num_inputs)
x3 = np.concatenate([x1, x2], axis=0)
#create activation function
relu_fn = lambda x: np.maximum(x, 0)
#create fully connected layer
FClayer = FullyConnectedLayer(num_inputs, layer_size, relu_fn)
#outputs
out1 = FClayer.forward(x1)
out2 = FClayer.forward(x2)
out3 = FClayer.forward(x3)
# print(out1.shape)
# print(out2)
# print(out3)



"""#######################
Page 36: MNIST handwritten number
#########################"""
from tensorflow.keras.datasets import mnist
# np.random.seed(42)
#Load the training and testing dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10
#Change inputs into column-vectors
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
y_train = np.eye(num_classes)[y_train]

#sigmoid function
sigmoid = lambda x: 1/(1+np.exp(-x))

#A simple FC neural network
class SimpleNetwork(object):
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32)):
        super().__init__()
        sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.Layers = [FullyConnectedLayer(sizes[i], sizes[i+1], sigmoid) for i in range(len(sizes)-1)]
    def forward(self, x):
        #forward the input x through all layers
        for layer in self.Layers:
            x = layer.forward(x)
        return x
    def predict(self, x):
        #Change output to number
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class
    def evaluate_accuracy(self, X_val, y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            if self.predict(X_val[i] == y_val[i]):
                num_corrects += 1
        return num_corrects/len(X_val)    


#Apply for MNIST images
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64, 32])
accuracy = mnist_classifier.evaluate_accuracy(X_test, y_test)
print("Accuracy = {:.8f}%".format(accuracy * 100))
