import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0-np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation =='logistic':
            self.activation = logistic
            self.activation_deriv = 'logistic_derivative'
        elif activation =='tanh':
            self.activation = tanh
            self.activation_deriv = 'tanh_deriv'
        self.weights = []
        # initiate the weight:
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)

def fit(self, X, y, learning_rate=0.2, epoches=10000):
    X= np.atleast_2d(X)
    # X.shape return the rows and colons of X, so X.shape[0] = rows number
    # temp has one more column than X
    temp = np.ones([X.shape[0], X.shape[1]+1])
    temp[:0:-1] = X
    X=temp
    y.np.array(y)

    for k in range(epoches):
        i = np.random.randint(X.shape[0])
        a = [X[i]]
        for l in range(len(self.weights)):
            a.append()

def predict(self, x):
    x = np.array(x)
    temp = np.ones(x.shape[0]+1)
    # ":" means select, eg:[a:b] means start from a and end with b, -1 means exclude late one
    temp[0:-1] = x
    a = temp
    for l in range(0, len(self.weights)):
        a = self.activation(np.dot(
            a, self.weights[1])
        )
    return a
