from model import *
import numpy as np
from matplotlib import pyplot as plt


# activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x);


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2;


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size;


def testing_model(x_train, y_train):
    # network
    net = Network()
    net.add(FCLayer(2, 2))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(2, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=2000, learning_rate=0.2, stop_correct=True)

    # test
    raw_out = net.predict(x_train)
    print([round(float(x), 0) for x in raw_out])


def check_activation_functions(number_of_iterations):
    number_of_epochs = 1000
    learning_rate = 0.2

    last_errors_sigmoid = []
    last_errors_tanh = []
    # checking how different activation function affect the training
    for i in range(number_of_iterations):
        weights_first = np.random.rand(2, 2) - 0.5
        bias_first = np.random.rand(1, 2) - 0.5
        weights_second = np.random.rand(2, 1) - 0.5
        bias_second = np.random.rand(1, 1) - 0.5

        net1 = Network()
        net1.add(FCLayer(2, 2, weights_first, bias_first))
        net1.add(ActivationLayer(sigmoid, sigmoid_prime))
        net1.add(FCLayer(2, 1, weights_second, bias_second))
        net1.add(ActivationLayer(sigmoid, sigmoid_prime))
        net1.use(mse, mse_prime)
        net1.fit(x_train, y_train, epochs=number_of_epochs, learning_rate=learning_rate, stop_correct=False)
        last_errors_sigmoid.append(net1.errors[-1])

        net2 = Network()
        net2.add(FCLayer(2, 2))
        net2.add(ActivationLayer(tanh, tanh_prime))
        net2.add(FCLayer(2, 1))
        net2.add(ActivationLayer(tanh, tanh_prime))
        net2.use(mse, mse_prime)
        net2.fit(x_train, y_train, epochs=number_of_epochs, learning_rate=learning_rate, stop_correct=False)
        last_errors_tanh.append(net2.errors[-1])

        # plt.plot([i for i in range(number_of_epochs)], net1.errors, label="sigmoid")
        # plt.plot([i for i in range(number_of_epochs)], net2.errors, label="tanh")
        # plt.title("Error over epochs")
        # plt.xlabel("Epochs")
        # plt.ylabel("Error")
        # plt.legend()
        # plt.show()

    print(f"Mean of last errors for sigmoid: {np.mean(last_errors_sigmoid)} and for tanh: {np.mean(last_errors_tanh)}")


def check_number_of_epochs():
    epochs = [i for i in range(100, 2000, 100)]
    last_errors = []
    weights_first = np.random.rand(2, 2) - 0.5
    bias_first = np.random.rand(1, 2) - 0.5
    weights_second = np.random.rand(2, 1) - 0.5
    bias_second = np.random.rand(1, 1) - 0.5

    for i in epochs:
        net = Network()
        net.add(FCLayer(2, 2, weights_first, bias_first))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.add(FCLayer(2, 1, weights_second, bias_second))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.use(mse, mse_prime)
        net.fit(x_train, y_train, epochs=i, learning_rate=0.2, stop_correct=False)
        last_errors.append(net.errors[-1])

    plt.plot(epochs, last_errors)
    plt.title("Last error over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


def check_stopping_criterion(number_of_iterations):
    no_stop_errors = []
    stop_errors = []
    stopped_after_epochs = []

    for i in range(number_of_iterations):
        weights_first = np.random.rand(2, 2) - 0.5
        bias_first = np.random.rand(1, 2) - 0.5
        weights_second = np.random.rand(2, 1) - 0.5
        bias_second = np.random.rand(1, 1) - 0.5

        net1 = Network()
        net1.add(FCLayer(2, 2, weights_first, bias_first))
        net1.add(ActivationLayer(sigmoid, sigmoid_prime))
        net1.add(FCLayer(2, 1, weights_second, bias_second))
        net1.add(ActivationLayer(sigmoid, sigmoid_prime))
        net1.use(mse, mse_prime)
        net1.fit(x_train, y_train, epochs=2000, learning_rate=0.2, stop_correct=False)

        net2 = Network()
        net2.add(FCLayer(2, 2, weights_first, bias_first))
        net2.add(ActivationLayer(sigmoid, sigmoid_prime))
        net2.add(FCLayer(2, 1, weights_second, bias_second))
        net2.add(ActivationLayer(sigmoid, sigmoid_prime))
        net2.use(mse, mse_prime)
        net2.fit(x_train, y_train, epochs=2000, learning_rate=0.2, stop_correct=True)

        no_stop_errors.append(net1.errors[-1])
        stop_errors.append(net2.errors[-1])
        stopped_after_epochs.append(net2.stopped_after_epochs)

    print(
        f"Mean of last errors for no stopping criterion: {np.mean(no_stop_errors)} and for stopping criterion: {np.mean(stop_errors)}")
    print(f"Mean of epochs for stopping criterion: {np.mean(stopped_after_epochs)}")


def check_learning_rate():
    learning_rate = [i for i in range(1, 20, 1)]
    last_errors = []
    weights_first = np.random.rand(2, 2) - 0.5
    bias_first = np.random.rand(1, 2) - 0.5
    weights_second = np.random.rand(2, 1) - 0.5
    bias_second = np.random.rand(1, 1) - 0.5

    for i in learning_rate:
        net = Network()
        net.add(FCLayer(2, 2, weights_first, bias_first))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.add(FCLayer(2, 1, weights_second, bias_second))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.use(mse, mse_prime)
        net.fit(x_train, y_train, epochs=2000, learning_rate=i / 10, stop_correct=False)
        last_errors.append(net.errors[-1])

    learning_rate = [float(i) / 10 for i in learning_rate]
    plt.plot(learning_rate, last_errors)
    plt.title("Last error over learning rate")
    plt.xlabel("Learning rate")
    plt.ylabel("Error")
    plt.show()


def chceck_momentum():
    momentum = [i for i in range(0, 10, 1)]
    weights_first = np.random.rand(2, 2) - 0.5
    bias_first = np.random.rand(1, 2) - 0.5
    weights_second = np.random.rand(2, 1) - 0.5
    bias_second = np.random.rand(1, 1) - 0.5
    last_errors = []

    for i in momentum:
        net = Network()
        net.add(FCLayer(2, 2, weights_first, bias_first))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.add(FCLayer(2, 1, weights_second, bias_second))
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
        net.use(mse, mse_prime)
        net.fit(x_train, y_train, epochs=2000, learning_rate=0.2, stop_correct=False, momentum=i / 10)
        last_errors.append(net.errors[-1])

    momentum = [float(i) / 10 for i in momentum]
    plt.plot(momentum, last_errors)
    plt.title("Last error over momentum")
    plt.xlabel("Momentum")
    plt.ylabel("Error")
    plt.show()


x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# testing_model(x_train, y_train)

# TODO: change functions to return (not display) results and (if possible) plot their means instead of individual
#  results
for i in range(1):
    # checking how different activation function affect the training
    check_activation_functions(10)
    # checking how number of epochs affects the training
    check_number_of_epochs()
    # checking how stopping criterion affects the training
    check_stopping_criterion(10)
    # checking how learning rate affects the training
    check_learning_rate()
    # checking how momentum affects the training
    chceck_momentum()
