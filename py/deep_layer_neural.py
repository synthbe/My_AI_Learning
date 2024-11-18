import pdb # pyright: ignore
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

# pyright: strict

np.random.seed(32)
FloatArray: TypeAlias = NDArray[np.float64]
DictParameters: TypeAlias = dict[str, FloatArray]
Cache: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]

class NeuralNet:
    parameters: DictParameters = dict()
    caches: list[Cache] = list()
    grads: DictParameters = dict()

    def __init__(self, topology: list[int]) -> None:
        L = len(topology)
        for i in range(1, L):
            self.parameters["W" + str(i)] = np.random.randn(topology[i], topology[i-1])
            self.parameters["b" + str(i)] = np.zeros((topology[i], 1))

    def __repr__(self) -> str:
        return f"parameters: {self.parameters} | Caches: {self.caches} | Gradiants: {self.grads}"


    def linear_forward(self, A: FloatArray, W: FloatArray, b: FloatArray) -> FloatArray:
        return np.dot(W, A) + b

    def relu_forward(self, Z: FloatArray) -> FloatArray:
        return np.maximum(0.0, Z)

    def sigmoid_forward(self, Z: FloatArray) -> FloatArray:
        return 1 / ( 1 + np.exp(-Z) )

    def forward_prop(self, X: FloatArray) -> FloatArray:
        self.caches.clear()
        A_prev = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            W, b = self.parameters['W' + str(l)], self.parameters['b' + str(l)]
            Z = self.linear_forward(A_prev, W, b)
            A = self.relu_forward(Z)

            self.caches.append((Z, A_prev, W))

            A_prev = A

        W, b = self.parameters['W' + str(L)], self.parameters['b' + str(L)]
        Z = self.linear_forward(A_prev, W, b)
        A_l = self.sigmoid_forward(Z)
        self.caches.append((Z, A_prev, W))

        return A_l

    def loss_function(self, y: FloatArray, a: FloatArray) -> FloatArray:
        epsilon = 1e-15
        a = np.clip(a, epsilon, 1 - epsilon)
        return (y * np.log(a) + (1 - y) * np.log(1 - a))

    def cost_function(self, y: FloatArray, a: FloatArray) -> float:
        m = y.shape[1]
        return -(1 / m) * np.sum(self.loss_function(y, a))

    def linear_backward(self, dZ: FloatArray, A: FloatArray, W: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        m = A.shape[1]

        dA = np.dot(W.T, dZ)
        dW = np.dot(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        return dA, dW, db

    def relu_back(self, Z: FloatArray) -> FloatArray:
        return np.maximum(0, (Z + 1) - Z)

    def sigmoid_back(self, Z: FloatArray) -> FloatArray:
        return self.sigmoid_forward(Z) * (1 - self.sigmoid_forward(Z))

    def back_prop(self, Y: FloatArray, A_l: FloatArray) -> None:

        L = len(self.caches)
        dA_l = -(np.divide(Y, A_l) - np.divide(1 - Y, 1 - A_l)) # Derivative of loss related to A_l
        Z, A, W = self.caches[-1]
        dZ = dA_l * self.sigmoid_back(Z)

        dA, dW, db = self.linear_backward(dZ, A, W)
        self.grads[f'dA{L-1}'] = dA
        self.grads[f'dW{L}'] = dW
        self.grads[f'db{L}'] = db

        for l in range(L-2, -1, -1):
            Z, A, W = self.caches[l]
            dZ = dA * self.relu_back(Z)
            dA, dW, db = self.linear_backward(dZ, A, W)
            self.grads[f'dA{l}'] = dA
            self.grads[f'dW{l+1}'] = dW
            self.grads[f'db{l+1}'] = db


    def update_parameters(self, learning_rate: float = 0.001) -> None:
        L: int = len(self.parameters) // 2
        for i in range(1, L + 1):
            self.parameters[f'W{i}'] -= learning_rate * self.grads[f'dW{i}']
            self.parameters[f'b{i}'] -= learning_rate * self.grads[f'db{i}']


EPOCHS: int = 20

if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])

    n_x, n_y = X.shape[0], y.shape[0] # input and output neurons shape

    net = NeuralNet([n_x, 32, 32, n_y])

    for _ in range(EPOCHS):
        al = net.forward_prop(X, )
        net.back_prop(y, al)
        net.update_parameters(learning_rate=0.01)

        print(net.cost_function(y, al))

    al = net.forward_prop(X,)

    [print(round(val)) for val in al[0,:]]

# Fix: Problem with large layers or large amount of layers
