import optimizer
from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt


def read_data():
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
    return x_train, y_train, x_validation, y_validation


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def loss_function(x, y, w):
    sigmoid_xw = sigmoid(x.dot(w))
    loss = -(np.dot(y.T, np.log(sigmoid_xw)) + (1 - y).T.dot(np.log(1 - sigmoid_xw)))[0]
    return loss / x.shape[0]


def gradient_function(x, y, w):
    return x.T.dot(sigmoid(np.dot(x, w)) - y) / x.shape[0]


def main():
    x_train, y_train, x_validation, y_validation = read_data()
    loss_validation = {'SGD': [], 'NAG': [], 'RMSProp': [], 'AdaDelta': [], 'Adam': []}

    #SGD
    SGD = optimizer.SGD(learning_rate=0.5)
    SGD.training_data(x_train, y_train)
    SGD.minimize(loss_function=loss_function, gradient_function=gradient_function)

    #NAG
    NAG = optimizer.NAG(learning_rate=0.08, gamma=0.9)
    NAG.training_data(x_train, y_train)
    NAG.minimize(loss_function=loss_function, gradient_function=gradient_function)

    #RMSProp
    RMSProp = optimizer.RMSProp(learning_rate=0.01, gamma=0.9, epsilon=1e-8)
    RMSProp.training_data(x_train, y_train)
    RMSProp.minimize(loss_function=loss_function, gradient_function=gradient_function)

    #AdaDelta
    AdaDelta = optimizer.AdaDelta(learning_rate=0.001, gamma=0.95, epsilon=0.001)
    AdaDelta.training_data(x_train, y_train)
    AdaDelta.minimize(loss_function=loss_function, gradient_function=gradient_function)

    #Adam
    Adam = optimizer.Adam(learning_rate=0.02, beta=0.9, gamma=0.999, epsilon=1e-5)
    Adam.training_data(x_train, y_train)
    Adam.minimize(loss_function=loss_function, gradient_function=gradient_function)

    for i in range(200):
        SGD.train()
        loss_validation['SGD'].append(SGD.loss(x_validation, y_validation))

        NAG.train()
        loss_validation['NAG'].append(NAG.loss(x_validation, y_validation))

        RMSProp.train()
        loss_validation['RMSProp'].append(RMSProp.loss(x_validation, y_validation))

        AdaDelta.train()
        loss_validation['AdaDelta'].append(AdaDelta.loss(x_validation, y_validation))

        Adam.train()
        loss_validation['Adam'].append(Adam.loss(x_validation, y_validation))

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