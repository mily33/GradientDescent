import numpy as np
import random


class Optimizer:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.w = None
        self.batch_size = None
        self.feature_num = None
        self.loss_function = None
        self.gradient_function = None
        self.predict_function = None

    def training_data(self, data_x, data_y, batch_size=0):
        self.x = data_x
        self.y = data_y
        self.batch_size = batch_size if batch_size != 0 else int(self.x.shape[0] / 6)
        self.feature_num = self.x.shape[1]

    def _samples(self):
        sample_index = random.sample(list(np.arange(0, self.x.shape[0])), self.batch_size)
        self.x_train = self.x[sample_index, :]
        self.y_train = self.y[sample_index, :]

    def loss(self, x, y):
        return self.loss_function(x, y, self.w)

    def predict(self, x):
        return self.predict_function(x, self.w)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.00001):
        super(SGD, self).__init__()
        self.learning_rate = learning_rate

    def minimize(self, loss_function, gradient_function, predict_function=None):
        self.w = np.random.randn(self.feature_num, 1)
        self.loss_function = loss_function
        self.gradient_function = gradient_function
        self.predict_function = predict_function

    def train(self):
        self._samples()
        delta = self.gradient_function(self.x_train, self.y_train, self.w)
        self.w = self.w - self.learning_rate * delta
        return self.loss_function(self.x, self.y, self.w)


class NAG(Optimizer):
    def __init__(self, learning_rate=0.00001, gamma=0.9):
        super(NAG, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v = None

    def minimize(self, loss_function, gradient_function, predict_function=None):
        self.w = np.random.randn(self.feature_num, 1)
        self.v = np.zeros((self.feature_num, 1))
        self.loss_function = loss_function
        self.gradient_function = gradient_function
        self.predict_function = predict_function

    def train(self):
        self._samples()
        g = self.gradient_function(self.x_train, self.y_train, self.w - self.gamma * self.v)
        self.v = self.gamma * self.v + self.learning_rate * g
        self.w = self.w - self.v
        return self.loss_function(self.x, self.y, self.w)


class RMSProp(Optimizer):
    def __init__(self, learning_rate = 0.00001, gamma=0.9, epsilon=1e-8):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = None

    def minimize(self, loss_function, gradient_function, predict_function=None):
        self.w = np.random.randn(self.feature_num, 1)
        self.G = np.zeros((self.feature_num, 1))
        self.loss_function = loss_function
        self.gradient_function = gradient_function
        self.predict_function = predict_function

    def train(self):
        self._samples()
        g = self.gradient_function(self.x_train, self.y_train, self.w)
        self.G = self.gamma * self.G + (1 - self.gamma) * g * g
        self.w = self.w - self.learning_rate / np.sqrt(self.G + self.epsilon) * g
        return self.loss_function(self.x, self.y, self.w)


class AdaDelta(Optimizer):
    def __init__(self, learning_rate=0.00001, gamma=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = None
        self.delta = None

    def minimize(self, loss_function, gradient_function, predict_function=None):
        self.w = np.random.randn(self.feature_num, 1)
        self.G = np.zeros((self.feature_num, 1))
        self.delta = np.zeros((self.feature_num, 1))
        self.loss_function = loss_function
        self.gradient_function = gradient_function
        self.predict_function = predict_function

    def train(self):
        self._samples()
        g = self.gradient_function(self.x_train, self.y_train, self.w)
        self.G = self.gamma * self.G + (1 - self.gamma) * g * g
        delta_w = - np.sqrt(self.delta + self.epsilon) / np.sqrt(self.G + self.epsilon) * g
        self.w = self.w + delta_w
        self.delta = self.gamma * self.delta + (1 - self.gamma) * delta_w * delta_w
        return self.loss_function(self.x, self.y, self.w)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.5, beta=0.9, gamma=0.95, epsilon=1e-8):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = None
        self.moments = None
        self.iteration = 1

    def minimize(self, loss_function, gradient_function, predict_function=None):
        self.w = np.random.randn(self.feature_num, 1)
        self.G = np.zeros((self.feature_num, 1))
        self.moments = np.zeros((self.feature_num, 1))
        self.loss_function = loss_function
        self.gradient_function = gradient_function
        self.predict_function = predict_function

    def train(self):
        self._samples()
        g = self.gradient_function(self.x_train, self.y_train, self.w)
        self.moments = self.beta * self.moments + (1.0 - self.beta) * g
        self.G = self.gamma * self.G + (1.0 - self.gamma) * g * g
        param1 = np.sqrt(1.0 - self.gamma ** self.iteration)
        param2 = (1.0 - self.beta ** self.iteration)
        alpha = self.learning_rate * param1 / param2
        self.w = self.w - alpha * self.moments / np.sqrt(self.G + self.epsilon)
        return self.loss_function(self.x, self.y, self.w)
