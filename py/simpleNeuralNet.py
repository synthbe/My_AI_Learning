import numpy as np

# Initialize data
inputs = np.array([[2, 5], [2, 7], [7, 11], [11, 17], [19, 23]], dtype=float)
labels = np.array([0, 0, 1, 0, 1], dtype=float)

# Preprocessing
inputs /= np.amax(inputs)
labels /= np.amax(labels)

# Neural Net 

input_layer = np.random.uniform(-0.5, -0.5, (3, 2))
input_bias = np.zeros((3, 1))
output_layer = np.random.uniform(-0.5, 0.5, (1, 3))
output_bias = np.zeros((1, 1))

epochs = 3
learning_rate = 0.01
error = 0

for epoch in range(epochs):
    for input, label in zip(inputs, labels):
        input = input.reshape(-1, 1)
        label = label.reshape(-1, 1)

        z1 = (input_layer @ input) + input_bias
        o1 = np.maximum(0, z1)

        z2 = (output_layer @ o1) + output_bias
        o2 = (1 + (1 / (np.exp(-z2))))

        error += np.mean((o2 - label)**2)

        error_grad = o2 - label

        weight_grad = error_grad @ o1.T
        bias_grad = error_grad * (1 - (1 + (1 / (np.exp(-z2)))))

        error_grad2 = error_grad * output_layer.T @ (1 - (1 + (1 / (np.exp(-z2))))) 

        output_layer -= learning_rate * weight_grad
        output_bias -= learning_rate * bias_grad

        weight_grad = 
    print(error)
    error = 0