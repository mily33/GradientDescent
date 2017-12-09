from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def loss(x, y, w):
    sigmoid_xw = sigmoid(x.dot(w))
    return -(np.dot(y.T, np.log(sigmoid_xw)) + (1 - y).T.dot(np.log(1 - sigmoid_xw)))[0]


def SGD(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.00001):
    loss_validation = []
    w_SGD = np.random.randn(x_train.shape[1], 1)

    for i in range(200):

        sample_index = random.sample(list(np.arange(0, x_train.shape[0])), batch_size)
        x = x_train[sample_index, :]
        y = y_train[sample_index, :]

        loss_validation.append(loss(x_validation, y_validation, w_SGD) / x_validation.shape[0])

        delta_SGD = x.T.dot(sigmoid(np.dot(x, w_SGD)) - y) / x.shape[0]
        w_SGD = w_SGD - learning_rate * delta_SGD

    return loss_validation


def NAG(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.00001, gamma=0.9):

    w_NAG = np.random.randn(x_train.shape[1], 1)
    v = np.zeros((x_train.shape[1], 1))

    loss_validation = []

    for i in range(200):
        sample_index = random.sample(list(np.arange(0, x_train.shape[0])), batch_size)
        x = x_train[sample_index, :]
        y = y_train[sample_index, :]

        loss_validation.append(loss(x_validation, y_validation, w_NAG) / x_validation.shape[0])

        g = x.T.dot(sigmoid(np.dot(x, w_NAG - gamma * v)) - y) / x.shape[0]
        v = gamma * v + learning_rate * g
        w_NAG = w_NAG - v

    return loss_validation


def RMSProp(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.00001, gamma=0.9, epsilon=1e-8):
    w_RMSProp = np.random.randn(x_train.shape[1], 1)
    G = np.zeros((x_train.shape[1], 1))

    loss_validation = []

    for i in range(200):
        sample_index = random.sample(list(np.arange(0, x_train.shape[0])), batch_size)
        x = x_train[sample_index, :]
        y = y_train[sample_index, :]

        loss_validation.append(loss(x_validation, y_validation, w_RMSProp) / x_validation.shape[0])

        g = x.T.dot(sigmoid(np.dot(x, w_RMSProp)) - y) / x.shape[0]
        G = gamma * G + (1 - gamma) * g * g
        w_RMSProp = w_RMSProp - learning_rate / np.sqrt(G + epsilon) * g

    return loss_validation


def AdaDelta(x_train, y_train, x_validation, y_validation, batch_size=5000, gamma=0.95, epsilon=1e-8):
    w_AdaDelta = np.random.randn(x_train.shape[1], 1)
    G = np.zeros((x_train.shape[1], 1))
    delta = np.zeros((x_train.shape[1], 1))
    loss_validation = []

    for i in range(200):
        sample_index = random.sample(list(np.arange(0, x_train.shape[0])), batch_size)
        x = x_train[sample_index, :]
        y = y_train[sample_index, :]

        loss_validation.append(loss(x_validation, y_validation, w_AdaDelta) / x_validation.shape[0])

        g = x.T.dot(sigmoid(np.dot(x, w_AdaDelta)) - y) / x.shape[0]
        G = gamma * G + (1 - gamma) * g * g
        delta_w = - np.sqrt(delta + epsilon) / np.sqrt(G + epsilon) * g
        w_AdaDelta = w_AdaDelta + delta_w
        delta = gamma * delta + (1 - gamma) * delta_w * delta_w

    return loss_validation


def Adam(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.5, beta=0.9, gamma=0.95, epsilon=1e-8):
    w_Adam = np.random.randn(x_train.shape[1], 1)
    G = np.zeros((x_train.shape[1], 1))
    moments = np.zeros((x_train.shape[1], 1))
    loss_validation = []

    for i in range(200):
        sample_index = random.sample(list(np.arange(0, x_train.shape[0])), batch_size)
        x = x_train[sample_index, :]
        y = y_train[sample_index, :]

        loss_validation.append(loss(x_validation, y_validation, w_Adam) / x_validation.shape[0])

        g = x.T.dot(sigmoid(np.dot(x, w_Adam)) - y) / x.shape[0]
        moments = beta * moments + (1.0 - beta) * g
        G = gamma * G + (1.0 - gamma) * g * g
        alpha = learning_rate * np.sqrt(1.0 - gamma**(i + 1)) / (1.0 - beta**(i + 1))
        w_Adam = w_Adam - alpha * moments / np.sqrt(G + epsilon)


    return loss_validation



def main():
    data_train = load_svmlight_file("/home/mily/a9a")
    data_validation = load_svmlight_file('/home/mily/a9a.t', n_features=123)

    x_train = data_train[0].toarray()
    y_train = data_train[1].reshape(x_train.shape[0], 1)

    x_validation = data_validation[0].toarray()
    y_validation = data_validation[1].reshape(x_validation.shape[0], 1)

    x_train = np.concatenate((np.ones((x_train.shape[0], 1), dtype='float'), x_train), axis=1)
    x_validation = np.concatenate((np.ones((x_validation.shape[0], 1), dtype='float'), x_validation), axis=1)

    y_train[y_train == -1] = 0
    y_validation[y_validation == -1] = 0

    loss_validation = {'SGD': [], 'NAG': [], 'RMSProp': [], 'AdaDelta': [], 'Adam': []}
    loss_validation['SGD'] = SGD(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.5)
    loss_validation['NAG'] = NAG(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.1, gamma=0.9)
    loss_validation['RMSProp'] = RMSProp(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.01, gamma=0.9, epsilon=1e-8)
    loss_validation['AdaDelta'] = AdaDelta(x_train, y_train, x_validation, y_validation, batch_size=5000, gamma=0.95, epsilon=0.001)
    loss_validation['Adam'] = Adam(x_train, y_train, x_validation, y_validation, batch_size=5000, learning_rate=0.2, beta=0.9, gamma=0.95, epsilon=1e-8)

    plot1, = plt.plot(np.arange(0, len(loss_validation['SGD'])), loss_validation['SGD'], 'b')
    plot2, = plt.plot(np.arange(0, len(loss_validation['NAG'])), loss_validation['NAG'], 'r')
    plot3, = plt.plot(np.arange(0, len(loss_validation['RMSProp'])), loss_validation['RMSProp'], 'g')
    plot4, = plt.plot(np.arange(0, len(loss_validation['AdaDelta'])), loss_validation['AdaDelta'], 'k')
    plot5, = plt.plot(np.arange(0, len(loss_validation['Adam'])), loss_validation['Adam'], 'y')

    plt.title('loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend([plot1, plot2, plot3, plot4, plot5], ['SGD', 'NAG', 'RMSProp', 'AdaDelta', 'Adam'])
    plt.show()


if __name__ == '__main__':
    main()
