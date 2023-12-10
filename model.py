import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass


class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, starting_weights=None, starting_bias=None):
        if starting_weights is None:
            self.weights = np.random.rand(input_size, output_size) - 0.5
        else:
            self.weights = starting_weights
        if starting_bias is None:
            self.bias = np.random.rand(1, output_size) - 0.5
        else:
            self.bias = starting_bias

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate, momentum_coef):
        # Compute input error
        input_error = np.dot(output_error, self.weights.T)

        # Compute gradients
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # Initialize momentum terms if not already done
        if not hasattr(self, 'weights_momentum'):
            self.weights_momentum = np.zeros_like(self.weights)
        if not hasattr(self, 'bias_momentum'):
            self.bias_momentum = np.zeros_like(self.bias)

        # Update momentum terms
        self.weights_momentum = momentum_coef * self.weights_momentum + weights_error
        self.bias_momentum = momentum_coef * self.bias_momentum + bias_error

        # Update parameters with momentum
        self.weights -= learning_rate * self.weights_momentum
        self.bias -= learning_rate * self.bias_momentum

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate, momentum_coef):
        return self.activation_prime(self.input) * output_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.errors = [0]
        self.stopped_after_epochs = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, stop_correct=False, momentum=0):
        self.stopped_after_epochs = epochs
        # sample dimension first
        samples = len(x_train)
        correct_out = [float(x) for x in y_train]

        # training loop
        for i in range(epochs):
            if stop_correct:
                raw_out = self.predict(x_train)
                out = [round(float(x), 0) for x in raw_out]

                if out == correct_out:
                    # print(f"Correct output after {i} epochs")
                    self.stopped_after_epochs = i
                    return

            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate, momentum)

            # calculate average error on all samples
            err /= samples
            self.errors.append(err)
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))