# # 1 1 --> 0
# # 1 0 --> 1
# # 0 1 --> 1
# # 0 0 --> 0
import numpy as np
import matplotlib.pylab as plt
from random import gauss

epochs = 50000

# target outputs
Y_16 = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]])

Y_32 = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                  0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]])

Y_64 = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                  0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                  0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                  0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]])
# learning rate
lr = 0.03

# standard deviation for noise
stdv = 0.5


def generate_noise_for_1():
    return gauss(1, stdv)


def generate_noise_for_0():
    return gauss(0, stdv)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def loss(Y, Ypred):
    return ((Y - Ypred) ** 2).mean()


def forward_prop(X, V, W):
     L1 = sigmoid(np.dot(X, V))
     L2 = sigmoid(np.dot(L1, W))
     return L2


def formulate(out):
    for i in range(len(out)):
        if out[i] >= 0.5:
            out[i] = 1
        else:
            out[i] = 0
    return out


# input 16
X_16 = np.array([[-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()]])

# input 32
X_32 = np.array([[-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()]])

# input 64
X_64 = np.array([[-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()]])

# input 64
X_64_test = np.array([
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()],
                 [-1, generate_noise_for_0(), generate_noise_for_0()],
                 [-1, generate_noise_for_0(), generate_noise_for_1()],
                 [-1, generate_noise_for_1(), generate_noise_for_0()],
                 [-1, generate_noise_for_1(), generate_noise_for_1()]])


class MLP4Neuron:
    def __init__(self):
        # weights in 4 neurons hidden layers, from -1 to 1
        self.V = np.random.random((3, 4)) * 2 - 1
        # weights in 4 neurons output layers, from -1 to 1
        self.W = np.random.random((4, 1)) * 2 - 1
        self.V_4 = None
        self.W_4 = None

    def init_weights(self):
        self.V_4 = self.V
        self.W_4 = self.W

    def forward(self, X):
        L1 = sigmoid(np.dot(X, self.V_4))
        L2 = sigmoid(np.dot(L1, self.W_4))
        return L2

    def draw_network(self):
        grayscale = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                grayscale[i][j] = self.forward([-1, 0.01 * i, 0.01 * j])

        plt.imshow(grayscale, cmap='gray')
        plt.show()

    def train_hidden4(self, X_train, Y_train):
        global lr
        # output from hidden layers 4x4 matrix
        L1 = sigmoid(np.dot(X_train, self.V_4))

        # output from output layers 4x1 matrix
        L2 = sigmoid(np.dot(L1, self.W_4))

        L2_delta = (Y_train.T - L2) * sigmoid_deriv(L2)

        L1_delta = L2_delta.dot(self.W_4.T) * sigmoid_deriv(L1)

        W_Change = lr * L1.T.dot(L2_delta)
        V_Change = lr * X_train.T.dot(L1_delta)

        self.W_4 = self.W_4 + W_Change
        self.V_4 = self.V_4 + V_Change

    def run_16(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden4(X_16, Y_16)
            if i % 500 == 0:
                output = forward_prop(X_16, self.V_4, self.W_4)
                MSE = loss(Y_16.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='4 hidden neurons, 16 training vectors')

    def run_32(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden4(X_32, Y_32)
            if i % 500 == 0:
                output = forward_prop(X_32, self.V_4, self.W_4)
                MSE = loss(Y_32.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='4 hidden neurons, 32 training vectors')

    def run_64(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden4(X_64, Y_64)
            if i % 500 == 0:
                output = forward_prop(X_64, self.V_4, self.W_4)
                MSE = loss(Y_64.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='4 hidden neurons, 64 training vectors')

    def generalisation_test(self):
        losses_train = []
        losses_test = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden4(X_64, Y_64)
            if i % 500 == 0:
                output_train = forward_prop(X_64, self.V_4, self.W_4)
                output_test = forward_prop(X_64_test, self.V_4, self.W_4)
                MSE_train = loss(Y_64.T, output_train)
                MSE_test = loss(Y_64.T, output_test)
                losses_train.append(MSE_train)
                losses_test.append(MSE_test)

        plt.plot(losses_train, label='4 hidden neurons, 64 training vectors, on training data')
        plt.plot(losses_test, label='4 hidden neurons, 64 training vectors, on test data')

class MLP2Neuron:
    def __init__(self):
        # weights in 2 neurons hidden layers, from -1 to 1
        self.V_copy = np.random.random((3, 2)) * 2 - 1
        # weights in 2 neurons output layers, from -1 to 1
        self.W_copy = np.random.random((2, 1)) * 2 - 1
        self.V_2 = None
        self.W_2 = None

    def init_weights(self):
        self.V_2 = self.V_copy
        self.W_2 = self.W_copy

    def forward(self, X):
        L1 = sigmoid(np.dot(X, self.V_2))
        L2 = sigmoid(np.dot(L1, self.W_2))
        return L2

    def draw_network(self):
        grayscale = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                grayscale[i][j] = self.forward([-1, 0.01 * i, 0.01 * j])

        plt.imshow(grayscale, cmap='gray')
        plt.show()

    def train_hidden2(self, X_train, Y_train):
        global lr
        # output from hidden layers 4x4 matrix
        L1 = sigmoid(np.dot(X_train, self.V_2))

        # output from output layers 4x1 matrix
        L2 = sigmoid(np.dot(L1, self.W_2))

        L2_delta = (Y_train.T - L2) * sigmoid_deriv(L2)

        L1_delta = L2_delta.dot(self.W_2.T) * sigmoid_deriv(L1)

        W_Change = lr * L1.T.dot(L2_delta)
        V_Change = lr * X_train.T.dot(L1_delta)

        self.W_2 = self.W_2 + W_Change
        self.V_2 = self.V_2 + V_Change

    def run_16(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden2(X_16, Y_16)
            if i % 500 == 0:
                output = forward_prop(X_16, self.V_2, self.W_2)
                MSE = loss(Y_16.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='2 hidden neurons, 16 training vectors')

    def run_32(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden2(X_32, Y_32)
            if i % 500 == 0:
                output = forward_prop(X_32, self.V_2, self.W_2)
                MSE = loss(Y_32.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='2 hidden neurons, 32 training vectors')

    def run_64(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden2(X_64, Y_64)
            if i % 500 == 0:
                output = forward_prop(X_64, self.V_2, self.W_2)
                MSE = loss(Y_64.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='2 hidden neurons, 64 training vectors')

    def generalisation_test(self):
        losses_train = []
        losses_test = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden2(X_64, Y_64)
            if i % 500 == 0:
                output_train = forward_prop(X_64, self.V_2, self.W_2)
                output_test = forward_prop(X_64_test, self.V_2, self.W_2)
                MSE_train = loss(Y_64.T, output_train)
                MSE_test = loss(Y_64.T, output_test)
                losses_train.append(MSE_train)
                losses_test.append(MSE_test)

        plt.plot(losses_train, label='2 hidden neurons, 64 training vectors, on training data')
        plt.plot(losses_test, label='2 hidden neurons, 64 training vectors, on test data')


class MLP8Neuron:
    def __init__(self):
        # weights in 8 neurons hidden layers, from -1 to 1
        self.V = np.random.random((3, 8)) * 2 - 1

        # weights in 8 neurons output layers, from -1 to 1
        self.W = np.random.random((8, 1)) * 2 - 1
        self.V_8 = None
        self.W_8 = None

    def init_weights(self):
        self.V_8 = self.V
        self.W_8 = self.W

    def forward(self, X):
        L1 = sigmoid(np.dot(X, self.V_8))
        L2 = sigmoid(np.dot(L1, self.W_8))
        return L2

    def draw_network(self):
        grayscale = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                grayscale[i][j] = self.forward([-1, 0.01 * i, 0.01 * j])

        plt.imshow(grayscale, cmap='gray')
        plt.show()

    def train_hidden8(self, X_train, Y_train):
        global lr
        # output from hidden layers 4x4 matrix
        L1 = sigmoid(np.dot(X_train, self.V_8))

        # output from output layers 4x1 matrix
        L2 = sigmoid(np.dot(L1, self.W_8))

        L2_delta = (Y_train.T - L2) * sigmoid_deriv(L2)

        L1_delta = L2_delta.dot(self.W_8.T) * sigmoid_deriv(L1)

        W_Change = lr * L1.T.dot(L2_delta)
        V_Change = lr * X_train.T.dot(L1_delta)

        self.W_8 = self.W_8 + W_Change
        self.V_8 = self.V_8 + V_Change

    def run_16(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden8(X_16, Y_16)
            if i % 500 == 0:
                output = forward_prop(X_16, self.V_8, self.W_8)
                MSE = loss(Y_16.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='8 hidden neurons, 16 training vectors')

    def run_32(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden8(X_32, Y_32)
            if i % 500 == 0:
                output = forward_prop(X_32, self.V_8, self.W_8)
                MSE = loss(Y_32.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='8 hidden neurons, 32 training vectors')

    def run_64(self):
        losses = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden8(X_64, Y_64)
            if i % 500 == 0:
                output = forward_prop(X_64, self.V_8, self.W_8)
                MSE = loss(Y_64.T, output)
                print("Iteration: " + str(i) + " MSE Error: " + str(MSE))
                losses.append(MSE)

        plt.plot(losses, label='8 hidden neurons, 64 training vectors')

    def generalisation_test(self):
        losses_train = []
        losses_test = []
        self.init_weights()
        for i in range(epochs):
            self.train_hidden8(X_64, Y_64)
            if i % 500 == 0:
                output_train = forward_prop(X_64, self.V_8, self.W_8)
                output_test = forward_prop(X_64_test, self.V_8, self.W_8)
                MSE_train = loss(Y_64.T, output_train)
                MSE_test = loss(Y_64.T, output_test)
                losses_train.append(MSE_train)
                losses_test.append(MSE_test)

        plt.plot(losses_train, label='8 hidden neurons, 64 training vectors, on training data')
        plt.plot(losses_test, label='8 hidden neurons, 64 training vectors, on test data')


mlp1 = MLP2Neuron()
mlp1.generalisation_test()
plt.legend()
plt.show()


