import typing as t
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

array = np.ndarray

def load_dataset() -> t.Tuple[array, array]:
    """
    Returning the dataset as a binary
    """

    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target

    X_binary = X[(y == '0') | (y == '1')]
    y_binary = y[(y == '0') | (y == '1')].astype(np.float32).values

    return X_binary, y_binary

def initialize_weights(dims: int) -> t.Tuple:
    """
    Weights must be a matrix of dims x 1, where dims is the size of the input.
    They can be initialized as zeros if prefered
    """

    weights: np.ndarray = np.random.randn(dims, 1)
    bias: float = 0

    return weights, bias

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)  # Clip values to prevent overflow
    return 1 / ( 1 + np.exp(-x) )

def loss(y, a):
    epsilon = 1e-15
    a = np.clip(a, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))

def propagate(X, y, w, b) -> t.Tuple[float, t.Dict[str, array]]:
    """
    This functions calculates the foward propagation and the backpropagation,
    returning the gradiants of the weights and bias and the cost value
    """

    x_pass = X.T
    y_pass = y.reshape(1, -1)
    m: int = X.shape[1]
    a: np.ndarray = np.zeros((1, m))
    z = np.dot(w.T, x_pass) + b # Matrix multiplication and broadcasting with numpy
    a = sigmoid(z)
    loss_val = loss(y_pass, a)
    cost = np.sum(loss_val, axis=1) / m

    # Equations for the backpropagation and the gradiants
    dz = a - y_pass
    dw = np.dot(x_pass, dz.T)
    db = np.sum(dz,) / m

    grads = {
        "dw": dw,
        "db": db
    }

    return cost, grads

def optimize(X, y, w, b, learning_rate=0.001) -> t.Tuple[array, float]:
    """
    This function executes the propagations and gets the gradiants to make the optimizations
    of the parameters, weights and bias
    """

    cost, grads = propagate(X, y, w, b)
    dw, db = grads["dw"], grads["db"]

    print(f"Current cost: {cost}")

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b

def predict(w, b, x_train, x_test, y_train, y_test) -> t.Tuple[t.Tuple[array, array], t.Tuple[array, array]]:
    """
    Function to make the predictions with already the trained values
    """

    x_train_pass, x_test_pass = x_train.T, x_test.T
    y_train_pass, y_test_pass = y_train.reshape(1, -1), y_test.reshape(1, -1)

    z = np.dot(w.T, x_train_pass) + b
    a = sigmoid(z)
    y_pred_train = np.array([[1 if i > 0.5 else 0 for i in a[0, :]]])

    z = np.dot(w.T, x_test_pass) + b
    a = sigmoid(z)
    y_pred_test = np.array([[1 if i > 0.5 else 0 for i in a[0, :]]])

    return (y_train_pass, y_test_pass), (y_pred_train, y_pred_test)

def acc_score(y_true, y_pred) -> float:
    n, m = len(y_true), len(y_pred)

    if n != m:
        return 0.0

    acc: int = 0
    for i, j in zip(y_true, y_pred):
        if i == j: acc += 1

    return acc / n

NUM_EPOCHS: int = 20

if __name__ == "__main__":
    X, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=32,)

    weights, bias = initialize_weights(X.shape[1])

    for i in range(NUM_EPOCHS):
        weights, bias = optimize(x_train, y_train, weights, bias)

    (y_train, y_test), (y_pred_train, y_pred_test) = predict(weights, bias, x_train, x_test, y_train, y_test)

    print(f"Accuracy in the training set: {acc_score(y_train[0], y_pred_train[0])}")
    print(f"Accuracy in the testing set: {acc_score(y_test[0], y_pred_test[0])}")
