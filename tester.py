from __future__ import print_function
import time
import numpy as np
# import tensorflow as tf # unused import?
from tensorflow import keras
from PIL import Image

ROUNDING = 6

# Prepare datasets: train, test, and our own
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()
X_TRAIN, X_TEST = X_TRAIN / 255.0, X_TEST / 255.0
print("MNIST data loaded")
X_OURS = []
Y_OURS = []
for image_index in range(10):
    image = Image.open("images/" + str(image_index) + ".jpg").convert('L')
    array = np.array(image)
    array = 1 - np.resize(array, ([28, 28]))
    X_OURS.append(array)
    Y_OURS.append(image_index)
X_OURS = np.array(X_OURS)
Y_OURS = np.array(Y_OURS)
print("Our data loaded")

# Hyperhyperparameters
ACTIVATORS = ["elu", "relu", "selu", "softmax", "softsign", "tanh", "hard_sigmoid", "sigmoid", "linear"]
OPTIMIZERS = ["sgd", "adamax", "adam", "nadam", "adadelta", "adagrad", "rmsprop"]
LOSSES = ["sparse_categorical_crossentropy"]

# Unclutters code, makes it easier to read
def is_int(num):
    try:
        int(num)
    except ValueError:
        return False
    return True
def is_float(num):
    try:
        float(num)
    except ValueError:
        return False
    return True

def train_mnist(name, layers, optimizer, learning_rate, loss, epochs, metrics, stoch_batch):

    start_time = time.time()
        
    # Define model
    print("Building model")
    model = keras.models.Sequential()
    layers_split = layers.lower().split("-")
    for layer in layers_split:

        # Assume flatten input
        if layer == "784":
            print("Assuming initial flatten layer")
            model.add(keras.layers.Flatten(input_shape=(28, 28)))

        # Raw numberical is dense layer
        elif is_int(layer):
            print("Dense layer recognized with hidden neurons=" + layer)
            model.add(keras.layers.Dense(int(layer)))

        # General activation functions
        elif layer in ACTIVATORS:
            print("Activation function recognized with function=" + layer)
            if layer == "exponential":
                model.add(keras.layers.Activation(activation=keras.activations.exponential))
            else:
                model.add(keras.layers.Activation(activation=layer))

        # Dropout layer
        elif "dropout" in layer:
            dropout_split = layer.split("_")
            if len(dropout_split) < 2 or not is_float(dropout_split[1]):
                print("Dropout layer recognized but invalid rate; skipping")
                continue
            dropout = float(dropout_split[1])
            if not 0 < dropout < 1:
                print("Dropout layer recognized but rate is not in range (0,1); skipping")
                continue
            print(("Dropout layer recognized with rate=" + dropout_split[1]))
            model.add(keras.layers.Dropout(float(dropout)))

        # L1 regularization
        elif "l1" in layer:
            dropout_split = layer.split("_")
            if len(dropout_split) < 2 or not is_float(dropout_split[1]):
                print("L1 reg layer recognized but invalid factor; skipping")
                continue
            print("L1 reg layer reocognized with factor=" + dropout_split[1])
            model.add(keras.layers.ActivityRegularization(l1=float(dropout_split)))

        # L2 regularization
        elif "l2" in layer:
            dropout_split = layer.split("_")
            if len(dropout_split) < 2 or not is_float(dropout_split[1]):
                print("L2 reg layer recognized but invalid factor; skipping")
                continue
            print("L2 reg layer reocognized with factor=" + dropout_split[1])
            model.add(keras.layers.ActivityRegularization(l2=float(dropout_split)))

        else:
            print("Unknown layer type: " + layer + ". Skipping")

    print("Model built")

    # Create and train model. Define custom evaluation
    print("Model fitting parameters: optimizer={0}, loss={1}, lr={2}".format(optimizer, loss, learning_rate))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    print("Beginning fit")
    history = model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, batch_size=stoch_batch)
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
    guess = model.predict(X_OURS)
    train_stats = evaluate_set(X_TRAIN, Y_TRAIN)
    train_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in train_stats.items()]
    test_stats = evaluate_set(X_TEST, Y_TEST)
    test_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in test_stats.items()]
    our_stats = evaluate_set(X_OURS, Y_OURS)
    our_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in our_stats.items()]
    err_loss = history.history["loss"][-1]

    # Output
    print("----------------------- TEST RESULTS ------------------------\n")
    print("Delta time (s): " + str(time.time() - start_time))
    print("Final training loss: " + str(round(err_loss, ROUNDING)))
    print()

    print("Train stats: " + " ".join(train_stats))
    print("Test stats: " + " ".join(test_stats))
    print("Our stats: " + " ".join(our_stats))
    print()

    print("Our predictions: " + str([np.argmax(x) for x in guess]))
    print("Full softmax:\n" + str(guess))


# Hyperparameters
OPTIMIZER = "adam"
LEARNING_RATE = 0
LOSS = "sparse_categorical_crossentropy"
EPOCHS = 2
METRICS = ['accuracy']
STOCH_BATCH = 256

# Layer definition
LAYERS = "784-500-relu-10-softmax"
"""
Each layer is separated by a dash.
Pure number is a simple linear layer
Within each layer string, an underscore (_) separates function from parameters
i.e. 'dropout_0.2' is 20% dropout
"""

train_mnist("Network:" + LAYERS, LAYERS, OPTIMIZER, LEARNING_RATE, LOSS, EPOCHS, METRICS, STOCH_BATCH)
