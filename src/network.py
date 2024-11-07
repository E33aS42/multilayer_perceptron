import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.scaler import Minmax_Scaler, Standard_Scaler
import pickle


class network():
    """
    A multilayer perceptron with custom hidden layers
    """

    def __init__(self, hidden_layers, lr, n_epochs,
                 init="random",
                 hidden_a="sigmoid",
                 output_a="softmax",
                 optim=("none", 0.9, 0.999),
                 prms={},
                 gradients={},
                 weighted_gradients={},
                 moving_average={},
                 patience=10,
                 early="no",
                 multi="no",
                 display="no"):
        """
                hidden_layers: tuple of number of neurons on each hidden layer
                lr: learning rate
                n_epochs: number of iterations
                init: initialization method: random or Xavier
                hidden_a: activation function for the hidden layers
                output_a: activation function for the output layer
                optim: optional opitmization algorithm
                display: display log loss and acccuracy at each epoch
                prms: a dictionary of the model parameters
                gradients: a dictionary of the log loss gradients
                weighted_gradients: weighted gradient saved during optiomization
                moving_average: moving average saved during optiomization
                patience: number of previous iterations used to evaluate early stopping
                early: set to "yes" to use early stopping otherwise "no"
                multi: set to "yes"for multiple trainings with different sets of prm (lr, hidden_a, output_a and/or optim)

        """
        try:
            self.hidden = hidden_layers
            self.lr = lr
            self.epochs = n_epochs
            self.prms = prms
            self.init = init
            self.hidden_a = hidden_a
            self.output_a = output_a
            self.optim = optim
            self.gradients = gradients
            self.weighted_gradients = weighted_gradients
            self.moving_average = moving_average
            self.patience = patience
            self.early = early
            self.mem_prms = [0] * patience
            self.mem_loss = [0] * patience
            self.multi = multi
            self.alpha = np.random.uniform(low=0.01, high=0.2)
            self.display = display
            self.beta = optim[1]
            self.beta2 = optim[2]
            print(
                f"\n\033[35m*** Model parameters ***\033[0m\n\
                -hidden_layers: {hidden_layers}\n\
                -lr: {lr}\n\
                -n_epochs: {n_epochs}\n\
                -init: {init}\n-hidden_a: {hidden_a}\n\
                -output_a: {output_a}\n-opti: {optim}\n\
                -early: {early}\n\
                -multi: {multi}\n\
                -patience: {patience}\n\
                -display: {display}\n")

        except Exception as e:
            print(e)
            return None


    def xavier_init(self, dim):
        """
        Xavier initialization:
        The range of random values for initializing weights is determined based
        on the number of input and output neurons in a layer.
        It aims to prevent the vanishing gradient problem.

        The vanishing gradient problem occurs when gradients,
        which are used to update the weights during backpropagation,
        become extremely small as they propagate through the layers.
        This results in slow  learning for earlier layers.
        """
        C = len(dim)
        np.random.seed(42)

        for c in range(1, C - 1):
            self.prms['W' + str(c)] = np.random.rand(dim[c],
                                                     dim[c - 1]) * np.sqrt(2. / dim[c])
            self.prms['b' + str(c)] = np.zeros((dim[c], 1))

        if self.output_a == "softmax":
            self.prms['W' + str(C - 1)] = np.random.rand(2 *
                                                         dim[C - 1], dim[C - 2]) * np.sqrt(2. / dim[c])
            self.prms['b' + str(C - 1)] = np.random.rand(2 * dim[C - 1], 1)
        else:
            self.prms['W' + str(C - 1)] = np.random.rand(dim[C - 1],
                                                         dim[C - 2]) * np.sqrt(2. / dim[c])
            self.prms['b' + str(C - 1)] = np.random.rand(dim[C - 1], 1)

        # Network layers topology:
        print("\n\033[33m*** Network layers topology ***\033[0m")
        for c in range(1, C):
            print(self.prms['W' + str(c)].shape)

        for c in range(1, C):
            self.weighted_gradients['dW' + str(c)] = 0
            self.weighted_gradients['db' + str(c)] = 0
            self.moving_average['dW' + str(c)] = 0
            self.moving_average['db' + str(c)] = 0


    def initialisation(self, dim):
        """
        random initialization
        """
        C = len(dim)

        np.random.seed(42)

        for c in range(1, C - 1):
            self.prms['W' + str(c)] = np.random.randn(dim[c],
                                                      dim[c - 1])
            self.prms['b' + str(c)] = np.random.randn(dim[c], 1)

        if self.output_a == "softmax":
            self.prms['W' + str(C - 1)] = np.random.randn(2 *
                                                          dim[C - 1], dim[C - 2])
            self.prms['b' + str(C - 1)] = np.random.randn(2 * dim[C - 1], 1)
        else:
            self.prms['W' + str(C - 1)
                      ] = np.random.randn(dim[C - 1], dim[C - 2])
            self.prms['b' + str(C - 1)] = np.random.randn(dim[C - 1], 1)

        # Network layers topology:
        print("\n\033[33m*** Network layers topology ***\033[0m")
        for c in range(1, C):
            print(self.prms['W' + str(c)].shape)

        for c in range(1, C):
            self.weighted_gradients['dW' + str(c)] = 0
            self.weighted_gradients['db' + str(c)] = 0
            self.moving_average['dW' + str(c)] = 0
            self.moving_average['db' + str(c)] = 0


    def relu(self, z):
        '''
        Return the rectified linear unit output of a vector z
        most commonly used activation function
        '''
        return np.maximum(0, z)


    def derivative_relu(self, a):
        (n, m) = a.shape
        return np.array([[1 if a[i, j] > 0 else 0 for j in range(m)] for i in range(n)])


    def leaky_relu(self, z):
        '''
        Return the leaky rectified linear unit output of a vector z
        using a randomly choosen leak paramater

        --> helps mitigate the "dying ReLU" problem,
        when the gradient becomes 0 if an input is negative.
        Weights will not be updated during training,
        reducing the model's learning capacity.
        '''
        return np.maximum(self.alpha * z, z)


    def derivative_leaky(self, a):
        (n, m) = a.shape
        return np.array([[1 if a[i, j] > 0 else self.alpha for j in range(m)] for i in range(n)])


    def tanh(self, z):
        '''
        Return the hyperbolic tangent output of a vector z
        more efficient than sigmoid
        '''
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return np.tanh(z)


    def derivative_tanh(self, a):
        return (1 - np.tanh(a) * np.tanh(a))


    def sigmoid(self, z):
        '''
        Return the sigmoid output of a vector z
        avoid using this except for the output layer
        '''
        return 1 / (1 + np.exp(-z))


    def derivative_sigmoid(self, a):
        return a * (1 - a)


    def softmax(self, z):
        '''
        Return the softmax output of a vector z
        '''
        exp_z = np.exp(z)
        epsilon = 1e-15
        sum_ = np.sum(exp_z, axis=0) + epsilon
        return exp_z / sum_


    def softmax1(self, z):
        '''
        Return the softmax output of a vector z for binary classification
        Equivalent to sigmoid function:
        softmax1 = 1 / (1 + exp(-2z))
        '''
        exp_z = np.exp(z)
        return exp_z / (exp_z + np.exp(-z))


    def batch_normalization(self, z):
        '''
        Apply minmax or zscore normalization
        '''
        my_Scaler = Minmax_Scaler()
        # my_Scaler = Standard_Scaler()
        my_Scaler.fit(z)
        return my_Scaler.transform(z)


    def forward_propagation(self, x):
        """
                feed forward propagation to calculate layers activations
        """

        # we initializae the activation dictionary with the input layer:
        activations = {'A0': x}
        # C is the last layer number of our network, or network length
        C = len(self.prms) // 2
        # note: the number of our network parameters is equal to twice the number of the network layers
        # to get the number of the last layer we then divide the number of parameters by 2 using the floor division to get an integer.

        # 	Hidden layers:
        for c in range(1, C):
            Z = self.prms['W' + str(c)].dot(activations['A' +
                                                        str(c - 1)]) + self.prms['b' + str(c)]

            # sigmoid activation
            if self.hidden_a == "sigmoid":
                activations['A' + str(c)] = self.sigmoid(Z)

            # tanh activation
            elif self.hidden_a == "tanh":
                activations['A' + str(c)] = self.tanh(Z)

            # relu activation
            elif self.hidden_a == "relu":
                activations['A' + str(c)] = self.relu(Z)

            # leaky relu activation
            elif self.hidden_a == "leaky":
                activations['A' + str(c)] = self.leaky_relu(Z)

        Z = self.prms['W' + str(C)].dot(activations['A' +
                                                    str(C - 1)]) + self.prms['b' + str(C)]

        # Output layer:
        # softmax activation
        if self.output_a == "softmax":
            activations['A' + str(C)] = self.softmax(Z)
        elif self.output_a == "softmax1":
            activations['A' + str(C)] = self.softmax1(Z)

            # sigmoid activation
        else:
            activations['A' + str(C)] = self.sigmoid(Z)

        return activations


    def back_propagation(self, y, activations):
        """
        Backward propagation to calculate gradients
        last layer: dZ = A(C) - y

        gradient for layer c:
        dJ/dW(c) = 1/m * dZ(c) * A(c - 1)

        dZ(c - 1) = W(c) * dZ(c) * dA(c - 1)/dZ(c - 1)
        """
        m = y.shape[1]
        C = len(self.prms) // 2
        dZ = activations['A' + str(C)] - y

        for c in range(C, 0, -1):
            self.gradients['dW' + str(c)] = 1/m * \
                np.dot(dZ, activations['A' + str(c - 1)].T)
            self.gradients['db' + str(c)] = 1/m * \
                np.sum(dZ, axis=1, keepdims=True)
            if c > 1:  # dZ doesnt exist if c <= 1
                if self.hidden_a == "sigmoid":
                    dZ = np.dot(self.prms['W' + str(c)].T, dZ) * \
                        self.derivative_sigmoid(activations['A' + str(c - 1)])
                elif self.hidden_a == "tanh":
                    dZ = np.dot(self.prms['W' + str(c)].T, dZ) * \
                        self.derivative_tanh(activations['A' + str(c - 1)])
                elif self.hidden_a == "relu":
                    dZ = np.dot(self.prms['W' + str(c)].T, dZ) * \
                        self.derivative_relu(activations['A' + str(c - 1)])
                elif self.hidden_a == "leaky":
                    dZ = np.dot(self.prms['W' + str(c)].T, dZ) * \
                        self.derivative_leaky(activations['A' + str(c - 1)])


    def update(self):
        """
        update gradient weights:
        W(t + 1) = W(t) - alpha * dJ/dW(t)
        """
        C = len(self.prms) // 2

        for c in range(1, C + 1):
            self.prms['W' + str(c)] = self.prms['W' + str(c)] - \
                self.lr * self.gradients['dW' + str(c)]
            self.prms['b' + str(c)] = self.prms['b' + str(c)] - \
                self.lr * self.gradients['db' + str(c)]


    def update_momentum(self):
        """
            Updating gradient weights applying momentum

        we are calculating an exponential weighted average of our gradients
        that we then use to update our weights instead.
        It may help dampen oscillations during gradient descent calculations.

        .How to calculate exponentially weighted average:
        This is a type of moving average that gives more weight to recent data points while applying exponentially decreasing weights to older observations.
        Vdw = beta * Vdw + (1 - beta) * dW
        Vdb = beta * Vdb + (1 - beta) * db

        dW could be interpreted as providing an acceleration, while Vdw could be thought as representing a velocity, and because beta is < to 1, it provides some friction that prevents going too fast.

        epsilon = 1E-8
        W -= alpha * VdW
        b -= alpha * Vdb

        beta is typically 0.9 which represents an averaging over the last 10 iterations gradients.

        """
        C = len(self.prms) // 2

        for c in range(1, C + 1):
            self.moving_average['dW' + str(c)] = self.beta * self.moving_average['dW' + str(c)] + (1 - self.beta) * self.gradients['dW' + str(c)]
            self.moving_average['db' + str(c)] = self.beta * self.moving_average['db' + str(c)] + (1 - self.beta) * self.gradients['db' + str(c)]
            self.prms['W' + str(c)] = self.prms['W' + str(c)] - \
                self.lr * self.moving_average['dW' + str(c)]
            self.prms['b' + str(c)] = self.prms['b' + str(c)] - \
                self.lr * self.moving_average['db' + str(c)]


    def update_RMSprop(self):
        """
            Updating gradient weights applying Root Mean Square propagation

        We are looking at an exponentially weighted average of the squares of the gradient:

        Sdw = beta * Sdw + (1 - beta) * (dW)**2
        Sdb = beta * Sdb + (1 - beta) * (db)**2

        It will dampen large gradients but amplify small gradients, smoothing the global weights update calculation.

        epsilon = 1E-8 (small number to avoid division by zero)
        W -= alpha * dW / sqrt(Sdw + epsilon)
        b -= alpha * db / sqrt(Sdb + epsilon)

        """
        C = len(self.prms) // 2
        epsilon = 1E-8

        for c in range(1, C + 1):
            self.weighted_gradients['dW' + str(c)] = self.beta * self.weighted_gradients['dW' + str(c)] + (1 - self.beta) * self.gradients['dW' + str(c)] * self.gradients['dW' + str(c)]
            self.weighted_gradients['db' + str(c)] = self.beta * self.weighted_gradients['db' + str(c)] + (1 - self.beta) * self.gradients['db' + str(c)] * self.gradients['db' + str(c)]

            self.prms['W' + str(c)] = self.prms['W' + str(c)] - self.lr * self.gradients['dW' + str(c)] / (np.sqrt(self.weighted_gradients['dW' + str(c)]) + epsilon)
            self.prms['b' + str(c)] = self.prms['b' + str(c)] - self.lr * self.gradients['db' + str(c)] / (np.sqrt(self.weighted_gradients['db' + str(c)]) + epsilon)


    def update_adam(self, i):
        """
            Adaptive Moment estimation.
            Updating gradient weights applying RMS propagation and momentum together

        . the 1st moment:
        Vdw = beta * Vdw + (1 - beta) * dW
        Vdb = beta * Vdb + (1 - beta) * db
        . The 2nd moment:
        Sdw = beta2 * Sdw + (1 - beta2) * (dW)**2
        Sdb = beta2 * Sdb + (1 - beta2) * (db)**2

        epsilon = 1E-8
        W -= alpha * VdW / sqrt(Sdw + epsilon)
        b -= alpha * Vdb / sqrt(Sdb + epsilon)

        """
        C = len(self.prms) // 2
        epsilon = 1E-8

        # bias correction (optional)
        if i >= 1:
            beta1cor = 1 - self.beta  # **i
            beta2cor = 1 - self.beta2  # **i
        else:
            beta1cor = 1
            beta2cor = 1

        for c in range(1, C + 1):
            # Momentum update
            self.moving_average['dW' + str(c)] =\
                (self.beta * self.moving_average['dW' + str(c)]
                    + (1 - self.beta) * self.gradients['dW' + str(c)])  # / beta1cor
            self.moving_average['db' + str(c)] =\
                (self.beta * self.moving_average['db' + str(c)]
                    + (1 - self.beta) * self.gradients['db' + str(c)])  # / beta1cor

            # RMSprop update
            self.weighted_gradients['dW' + str(c)] =\
                (self.beta2 * self.weighted_gradients['dW' + str(c)]
                    + (1 - self.beta2) * self.gradients['dW' + str(c)]
                    * self.gradients['dW' + str(c)])  # / beta2cor
            self.weighted_gradients['db' + str(c)] =\
                (self.beta2 * self.weighted_gradients['db' + str(c)]
                    + (1 - self.beta2) * self.gradients['db' + str(c)]
                    * self.gradients['db' + str(c)])  # / beta2cor

            # weights update
            self.prms['W' + str(c)] = self.prms['W' + str(c)]\
                - self.lr * self.moving_average['dW' + str(c)]\
                / np.sqrt(self.weighted_gradients['dW' + str(c)] + epsilon)
            self.prms['b' + str(c)] = self.prms['b' + str(c)]\
                - self.lr * self.moving_average['db' + str(c)]\
                / np.sqrt(self.weighted_gradients['db' + str(c)] + epsilon)


    def predict(self, x):
        """
        Make a prediction based on the probability a given data point belong to the class.
        1. binary classification:
            Given 2 classes 0 and 1, the output layer activation function
            (sigmoid) returns a probability that the data point
            belongs to class 1.
            If the found probability is equal or higher than 50%,
            we consider that the data point belongs to class 1,
            otherwise it belongs to class 0.

        2. multi-class classification:
            Given n classes, the output layer activation function (softmax)
            return n probabilities that a given data point belongs to
            a certain class versus the other n - 1 classes.
            The actual predicted class the data point belongs
            to is chosen by taking the class with the highest probability
            among those n probabilities.
        """
        activations = self.forward_propagation(x)
        C = len(self.prms) // 2
        A_train = activations['A' + str(C)]
        # print("A_train:", A_train.shape)
        if self.output_a == "softmax":
            y_pred = np.argmax(A_train, axis=0, keepdims=True)
            return y_pred
        else:
            return A_train >= 0.5

    def log_loss_(self, y, A):
        epsilon = 1e-15
        n = y.shape[0]
        m = y.shape[1]
        if n == 1:
            y = y.flatten()
            A = A.flatten()
            return 1 / len(y) * np.sum(-y * np.log(epsilon + A) - (1 - y) * np.log(epsilon + 1 - A))
        else:
            loss = - 1 / m * np.sum(y * np.log(epsilon + A))
            return loss


    def accuracy_(self, y, y_pred):
        n = len(y)
        acc = 0
        n = y.shape[0]
        m = y.shape[1]
        if n == 1:
            y = y.flatten()
            y_pred = y_pred.flatten()
            for i in range(m):
                if y[i] == y_pred[i]:
                    acc += 1
            return acc / m
        else:
            pass


    def MSE(self, y, y_hat):
        y = y.reshape(-1)
        y_hat = y_hat.reshape(-1)
        loss_elem_ = np.array([(yhi - yi)**2 for yhi, yi in zip(y_hat, y)])
        return float(sum(loss_elem_) / y.shape[0])


    def RMSE(self, y, y_hat):
        return np.sqrt(self.MSE(y, y_hat))


    def moveup(self, new_val, list_):
        """
        Move up list_ content values and replace the last item by new_val
        """
        for i in range(self.patience - 1):
            list_[i] = list_[i + 1]
        list_[-1] = new_val
        return list_


    def diff_mem_loss(self):
        """
        Calculate the difference between 2 consecutive iteration losses for the last self.patience iterations
        """
        diff_ = [None] * (self.patience - 1)
        for i in range(self.patience - 1):
            diff_[i] = self.mem_loss[i] - self.mem_loss[i + 1]
        return sum(diff_)


    def early_stopping(self, val_loss):
        """
        detect if validation loss starts increasing
        """
        self.moveup(self.prms, self.mem_prms)
        self.moveup(val_loss, self.mem_loss)
        diff_loss = self.diff_mem_loss()
        if diff_loss < 0:
            return 1
        return 0


    def get_y_labels(self, y):
        """
        return a list of the different labels found in array y of dimension (1, m)
        """
        labels = []
        for yi in y[0]:
            if yi not in labels:
                labels.append(yi)
        labels.sort()
        return labels


    def create_dict_labels(self, labels):
        """
        return a a dictionary labels key and corresponding int value
        """
        dict_labels = {}
        j = 0
        for i in labels:
            dict_labels[i] = j
            j += 1
        return dict_labels


    def make_y_softmax(self, y):
        """
        return an output array of labels for software activation
        input y must be a single column array of dimension (1, m)
        """
        try:
            assert y.shape[0], "input must be a single column array of dimension (1, m)"
            labels = self.get_y_labels(y)

            n = len(labels)
            m = y.shape[1]
            dict_labels = self.create_dict_labels(labels)
            y_softmax = np.zeros((n, m))
            i = 0
            for yi in y[0]:
                y_softmax[dict_labels[yi], i] = 1
                i += 1
            return y_softmax

        except Exception as e:
            print(e)
            exit()


    def training_loop(self, i, inputs, metrics):
        # get last layer number
        C = len(self.prms) // 2

        # recover inputs
        (x, y, x_val, y_val) = inputs
        (train_loss, train_acc, train_mse, train_rmse,
         valid_loss, valid_acc, valid_mse, valid_rmse) = metrics

        # 1-hot encoding for multi-class classification (softmax)
        y_hot = self.make_y_softmax(y)
        y_val_hot = self.make_y_softmax(y_val)

        ### loop content ###
        # training set
        activations = self.forward_propagation(x)
        if self.output_a == "softmax":
            self.back_propagation(y_hot, activations)
        else:
            self.back_propagation(y, activations)
        if self.optim[0] == "rms":
            self.update_RMSprop()
        elif self.optim[0] == "mom":
            self.update_momentum()
        elif self.optim[0] == "adam":
            self.update_adam(i)
        else:
            self.update()
        A_train = activations['A' + str(C)]
        # validation set
        A_val = self.forward_propagation(x_val)['A' + str(C)]

        ## log loss and accuracy calculation
        # training set
        if self.output_a == "softmax":
            train_loss.append(self.log_loss_(y_hot, A_train))
        else:
            train_loss.append(self.log_loss_(y, A_train))
        y_pred = self.predict(x)
        train_acc.append(self.accuracy_(y, y_pred))
        train_mse.append(self.MSE(y, y_pred))
        train_rmse.append(self.RMSE(y, y_pred))
        # validation set
        A_val = self.forward_propagation(x_val)['A' + str(C)]
        if self.output_a == "softmax":
            valid_loss.append(self.log_loss_(y_val_hot, A_val))
        else:
            valid_loss.append(self.log_loss_(y_val, A_val))
        y_val_pred = self.predict(x_val)
        valid_acc.append(self.accuracy_(y_val, y_val_pred))
        valid_mse.append(self.MSE(y_val, y_val_pred))
        valid_rmse.append(self.RMSE(y_val, y_val_pred))

        return (train_loss, train_acc, train_mse, train_rmse, valid_loss, valid_acc, valid_mse, valid_rmse)


    def plot_curves(self, metrics):
        (train_loss, train_acc, train_mse, train_rmse,
         valid_loss, valid_acc, valid_mse, valid_rmse) = metrics

        # 4 figures plot
        # plt.figure(figsize=(18, 4))
        # plt.subplot(1, 4, 1)
        # plt.plot(train_loss, linestyle='--', label='train loss')
        # plt.plot(valid_loss, linestyle='solid', label='valid loss')
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()
        # plt.subplot(1, 4, 2)
        # plt.plot(train_acc, linestyle='--', label='train acc')
        # plt.plot(valid_acc, linestyle='solid', label='valid acc')
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()
        # plt.subplot(1, 4, 3)
        # plt.plot(train_mse, linestyle='--', label='train MSE')
        # plt.plot(valid_mse, linestyle='solid', label='valid MSE')
        # plt.xlabel("Epochs")
        # plt.ylabel("MSE")
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()
        # plt.subplot(1, 4, 4)
        # plt.plot(train_rmse, linestyle='--', label='train RMSE')
        # plt.plot(valid_rmse, linestyle='solid', label='valid RMSE')
        # plt.xlabel("Epochs")
        # plt.ylabel("RMSE")
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()

         # 2 figures plot
        print(
            f"\n\033[32m*** Training results ***\033[0m\nTraining set - logsloss: {train_loss[-1]:0.4f}, accuracy: {train_acc[-1]:0.4f}\n")
        print(
            f"Validation set - logsloss: {valid_loss[-1]:0.4f}, accuracy: {valid_acc[-1]:0.4f}\n")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, linestyle='--', label='train loss')
        plt.plot(valid_loss, linestyle='solid', label='valid loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, linestyle='--', label='train acc')
        plt.plot(valid_acc, linestyle='solid', label='valid acc')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    def neural_network(self, x, y, x_val, y_val):
        # initialisation parameters
        dim = list(self.hidden)
        dim.insert(0, x.shape[0])
        dim.append(y.shape[0])
        # np.random.seed(1)
        if self.init == "xavier":
            print("xav")
            self.xavier_init(dim)
        else:
            self.initialisation(dim)

        # Lists storing various metrics
        train_loss = []
        train_acc = []
        train_mse = []
        train_rmse = []
        valid_loss = []
        valid_acc = []
        valid_mse = []
        valid_rmse = []
        metrics = (train_loss, train_acc, train_mse, train_rmse,
                   valid_loss, valid_acc, valid_mse, valid_rmse)
        inputs = (x, y, x_val, y_val)

        # 1-hot encoding for multi-class classification (softmax)
        y_hot = self.make_y_softmax(y)
        y_val_hot = self.make_y_softmax(y_val)

        # Save model topology
        print(f"\nx training shape: {x.shape}")
        print(f"x validation shape: {x_val.shape}")
        if self.output_a == "softmax":
            dim[-1] = y_hot.shape[0]
            print(f"y training shape: {y_hot.shape}")
            print(f"y validation shape: {y_val_hot.shape}\n")
        else:
            print(f"y training shape: {y.shape}")
            print(f"y validation shape: {y_val.shape}\n")

        # main iteration loop
        if self.display == "yes":
            for i in range(self.epochs):
                (train_loss, train_acc, train_mse, train_rmse, valid_loss, valid_acc,
                 valid_mse, valid_rmse) = self.training_loop(i, inputs, metrics)
                print(
                    f"epoch {i}/{self.epochs} - train.loss: {train_loss[-1]:.4f} - valid.loss: {valid_loss[-1]:.4f} - train.rmse: {train_rmse[-1]:.4f} - valid.rmse: {valid_rmse[-1]:.4f}")

                # early stopping
                self.early_stopping(valid_loss[-1])
                if i > 10 * self.patience and self.early == "yes" and self.early_stopping(valid_loss[-1]):
                    break
        else:
            for i in tqdm(range(self.epochs)):
                (train_loss, train_acc, train_mse, train_rmse, valid_loss, valid_acc,
                 valid_mse, valid_rmse) = self.training_loop(i, inputs, metrics)

                # early stopping
                self.early_stopping(valid_loss[-1])
                if i > 10 * self.patience and self.early == "yes" and self.early_stopping(valid_loss[-1]):
                    break

            if self.early == "yes":
                print(f"Current timestep before early stopping: {i}\n")

        # Learning curve plot
        print("\n\033[32m*** Training results ***\033[0m")
        print(f"* Training set *\n\
            logsloss: {train_loss[-1]:0.4f}\n\
            accuracy: {train_acc[-1]:0.4f}\n\
            MSE: {train_mse[-1]:0.4f}\n\
            RMSE: {train_rmse[-1]:0.4f}\n\
            ")
        print(f"* Validation set *\n\
        logsloss: {valid_loss[-1]:0.4f}\n\
        accuracy: {valid_acc[-1]:0.4f}\n\
        MSE: {valid_mse[-1]:0.4f}\n\
        RMSE: {valid_rmse[-1]:0.4f}\n\
        ")

        if self.multi == "yes":
            # multiplot with various validation loss curves for different training set parameters
            plt.plot(valid_loss, linestyle='--',
                     label=f'hid.layers: {self.hidden}, lr: {self.lr}, hid.activ.: {self.hidden_a}, out.activ.:{self.output_a}, optim.: {self.optim}')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")

        else:
            # Learning curve plots for one training set parameters
            self.plot_curves(metrics)

            # Save model
            with open("nn_model.pkl", "wb") as file:
                pickle.dump([self.hidden, self.lr, self.epochs, self.prms, self.init, self.hidden_a,
                            self.output_a, self.optim, self.gradients, self.weighted_gradients, self.moving_average], file)


    def save_csv(arr, name):
        df = pd.DataFrame(arr)
        # Save the DataFrame to a CSV file
        df.to_csv("data/" + name + ".csv", header=None, index=False)
        print(name + " file saved")


    def prediction(self, x_test, y_test):
        # Test set accuracy and log_loss
        test_loss = []
        test_acc = []

        C = len(self.prms) // 2
        y_test_hot = self.make_y_softmax(y_test)

        # test predictions
        A_test = self.forward_propagation(x_test)['A' + str(C)]
        if self.output_a == "softmax":
            test_loss.append(self.log_loss_(y_test_hot, A_test))
        else:
            test_loss.append(self.log_loss_(y_test, A_test))
        y_test_pred = self.predict(x_test)
        test_acc.append(self.accuracy_(y_test, y_test_pred))

        network.save_csv(y_test_pred, "y_test_pred")
        print(
            f"\033[34m*** Test results ***\033[34m\nLog loss: {test_loss[0]:.4f}\nAccuracy: {self.accuracy_(y_test, y_test_pred):.4f}")
