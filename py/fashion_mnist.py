from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Normalizando as imagens (Deixar os valores entre 0 e 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

modelo = keras.Sequential([ # Sequential define uma sequencia de camadas na rede
    keras.layers.Flatten(input_shape=(28, 28)), # Sao imagens 28x28, Flatten transforma a imagem em um array
    keras.layers.Dense(units=128, activation='relu'), 
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=10, activation='softmax') # 10 neuronios de saída para os 10 tipos de imagem
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modelo.fit(train_images, train_labels, epochs=5)
modelo.evaluate(test_images, test_labels)
