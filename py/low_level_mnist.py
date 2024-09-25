from tensorflow import keras
import numpy as np

def Relu(x):
    return np.maximum(0, x)

def derivativeRelu(x):
    return np.where(x < 0, 0, 1)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivativeSigmoid(x):
    return 1 - (Sigmoid(x))

def lossFunction(output, label):
    return np.mean((output - label)**2)

def derivativeLoss(output, label):
    return np.mean(output - label)

def initializeWeights(quantityOfLayers: int):
    hiddenWeights = list()

    inputWeights = np.random.uniform(-0.5, 0.5, (16, 784))

    for _ in range(quantityOfLayers):
        hiddenWeight = np.random.uniform(-0.5, 0.5, (16, 16))
        hiddenWeights.append(hiddenWeight)
    
    outputWeights = np.random.uniform(-0.5, 0.5, (10, 16))

    return inputWeights, hiddenWeights, outputWeights

def initializeBiases(quantity: int):
    hiddenBiases = list()
    inputBiases = np.zeros((16, 1))
    for _ in range(quantity):
        hiddenBias = np.zeros((16, 1))
        hiddenBiases.append(hiddenBias)

    outputBias = np.zeros((10, 1))

    return inputBiases, hiddenBiases, outputBias

def getDataset():
    mnist = keras.datasets.mnist

    (images_train, labels_train), (images_test, labels_test) = mnist.load_data()

    # Normalizing
    images_train, images_test = images_train / 255.00, images_test / 255.00 

    # One hot encoded
    ohe_labels_train = np.zeros((len(labels_train), 10))
    ohe_labels_train[np.arange(len(labels_train)), labels_train] = 1

    ohe_labels_test = np.zeros((len(labels_test), 10))
    ohe_labels_test[np.arange(len(labels_test)), labels_test] = 1

    return (images_train, ohe_labels_train), (images_test, ohe_labels_test)

learning_rate = 0.01
epochs = 5
accuracy, loss = 0, 0

if __name__ == "__main__":
    (images_train, labels_train), (images_test, labels_test) = getDataset()
    inputWeights, hiddenWeights, outputWeights = initializeWeights(3)
    inputBiases, hiddenBiases, outputBias = initializeBiases(3)

    for epoch in range(epochs):
        for image, label in zip(images_train, labels_train):
            image = image.reshape(-1, 1)
            label = label.reshape(-1, 1)

            # FeedFoward
            inputOutput = (inputWeights @ image) + inputBiases
            hiddenOutput = Relu(inputOutput)
            for weight, bias in zip(hiddenWeights, hiddenBiases):
                hiddenOutput = (weight @ hiddenOutput) + bias
                hiddenOutput = Relu(hiddenOutput)
            preoutput = (outputWeights @ hiddenOutput) + outputBias
            output = Sigmoid(preoutput)

            loss += lossFunction(output, label)

            # BackPropagation
            errorGrad = output - label
            weightGrad = errorGrad * derivativeSigmoid(output) @ preoutput.T
            biasGrad = errorGrad * derivativeSigmoid(output) 
            outputWeights -= learning_rate * weightGrad
            outputBias -= learning_rate * biasGrad

            errorGrad *= weight.T @ derivativeSigmoid(preoutput) 
            for weight, bias in reversed(zip(hiddenWeights, hiddenBiases)):
                weightGrad = errorGrad * derivativeRelu(output) @ hiddenOutput.T
                biasgrad = errorGrad * derivativeRelu(output)
                weight -= learning_rate * weightGrad
                bias -= learning_rate * biasGrad

                errorGrad *= weight.T @ derivativeRelu(output)
            
        print(loss)
        loss = 0


    #
    #
    #
    # for epoch in range(epochs):
    #     for img, label in zip(images_train, ohe_labels_train):
    #         img = img.reshape(-1, 1)
    #         label = label.reshape(-1, 1)
    #
    #         # Foward propagation
    #
    #         hidd_pre = (inp_hidd @ img) + bias_hidd # Input to Hidden
    #         # hidd = np.maximum(0, hidd_pre) # Relu
    #         hidd = 1 / (1 + np.exp(-hidd_pre))
    #
    #         output_pre = (hidd_out @ hidd) + bias_out # Hidden to Output
    #         output = 1 / (1 + np.exp(-output_pre))
    #
    #         # Loss Function
    #
    #         error += np.mean((output - label)**2) # Mean Squared Error
    #         shot += int(np.argmax(output) == np.argmax(label))
    #         
    #         # Back Propagation
    #
    #         delta = output - label
    #         hidd_out += -learning_rate * delta @ np.transpose(hidd)
    #         bias_out += -learning_rate * delta
    #
    #         delta_hidd = np.transpose(hidd_out) @ delta * (hidd * (1 - hidd))
    #         inp_hidd += -learning_rate * delta_hidd @ np.transpose(img)
    #         bias_hidd += -learning_rate * delta_hidd
    #
    #     print(f"Current Epoch: {epoch}")
    #     print(f"Acc: {round((shot / images_train.shape[0]) * 100, 2)}%")
    #     print(f"Loss: {error}")
    #     shot, error = 0, 0
