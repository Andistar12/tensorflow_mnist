from __future__ import print_function
import time
import numpy as np
from tf import keras as keras
from PIL import Image


# Prepare datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("MNIST data loaded")
ourdata_x = []
ourdata_y = []
for i in range(10):
    image = Image.open("images/" + str(i) + ".jpg").convert('L')
    array = np.array(image)
    array = 1 - np.resize(array, ([28,28]))
    ourdata_x.append(array)
    ourdata_y.append(i)
x_ours = np.array(ourdata_x)
y_ours = np.array(ourdata_y)
print("Our data loaded")

# Hyperhyperparameters
ACTIVATORS = ["elu", "relu", "selu", "softmax", "softsign", "tanh", "hard_sigmoid", "sigmoid", "exponential", "linear"]
OPTIMIZERS = ["sgd", "adamax", "adam", "nadam", "adadelta", "adagrad", "rmsprop"]

# Hyperparameters
start_time = time.time()
OPTIMIZER = "sgd"
LEARNING_RATE = 0
LOSS = "sparse_categorical_crossentropy"
EPOCHS = 500
STOCH_BATCH = 256
ROUNDING = 6

# Layer definition
LAYERS="784-500-sigmoid-10-softmax"
"""
Each layer is separated by a dash. 
Pure number is a simple linear layer
Within each layer string, a dot (.) separates function from parameters
i.e. 'dropout.0.2' is 20% dropout
"""

# Define model
print("Building model")
model = keras.models.Sequential()
layers_split = LAYERS.lower().split("-")
for layer in layers_split:

    # Assume flatten input
    if layer is "784":
        print("Assuming initial flatten layer")
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Raw numberical is dense layer
    elif isinstance(layer, int):
        print("Dense layer recognized with hidden neurons=" + layer)
        model.add(keras.layers.Dense(int(layer)))

    # General activation functions
    elif layer in ACTIVATORS:
        print("Activation function recognized with function=" + layer)
        model.add(keras.layers.Activation(activation=layer))

    # Dropout layer
    elif "dropout" in layer:
        dropout_split = layer.split(".")
        if len(dropout_split) < 2 or not isinstance(dropout_split[1], float):
            print("Dropout layer recognized but invalid rate; skipping")
            continue
        if not 0 < dropout_split[1] < 1:
            print("Dropout layer recognized but rate is not in range (0,1); skipping")
            continue
        print(("Dropout layer recognized with rate=" + dropout_split[1]))
        model.add(keras.layers.Dropout(float(dropout_split[1])))
    
    # L1 regularization
    elif "l1" in layer:
        dropout_split = layer.split(".")
        if len(dropout_split) < 2 or not isinstance(dropout_split[1], float):
            print("L1 reg layer recognized but invalid factor; skipping")
            continue
        print("L1 reg layer reocognized with factor=" + dropout_split[1])
        model.add(keras.layers.ActivityRegularization(l1=float(dropout_split)))

    # L2 regularization
    elif "l2" in layer:
        dropout_split = layer.split(".")
        if len(dropout_split) < 2 or not isinstance(dropout_split[1], float):
            print("L2 reg layer recognized but invalid factor; skipping")
            continue
        print("L2 reg layer reocognized with factor=" + dropout_split[1])
        model.add(keras.layers.ActivityRegularization(l2=float(dropout_split)))

print("Model built")


# Create and train model. Define custom evaluation
print("Model fitting parameters: optimizer={0}, loss={1}".format(OPTIMIZER, LOSS))
model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=['accuracy'])
print("Beginning fit")
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=STOCH_BATCH)
def evaluate_set(x_set, y_set):
    scores = model.evaluate(x_set, y_set)
    statistics = dict()
    for i in range(len(metrics)):
        key = model.metrics_names[i + 1]
        value = scores[i + 1] 
        statistics[key] = value
    return statistics
print("Training finished. Gathering data")

# Gather evaluation statistics
guess = model.predict(x_ours)
train_stats = evaluate_set(x_train, y_train)
test_stats = evaluate_set(x_test, y_test)
our_stats = evaluate_set(x_ours, y_ours)
err_loss = history.history["loss"][-1]

# Output
print("----------------------- TEST RESULTS ------------------------\n")
print("Delta time (s): " + str(time.time() - start_time))
print("Final training loss: " + str(round(err_loss, ROUNDING)))
print()

print("Train stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in train_stats.items()]))
print("Test stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in test_stats.items()]))
print("Our stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in our_stats.items()]))
print()

print("Our predictions: " + str([np.argmax(x) for x in guess]))
print("Full softmax:\n" + str(guess))
