from __future__ import print_function
import time
import numpy as np
# import tensorflow as tf # unused import?
from tensorflow import keras
from PIL import Image


# Prepare datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("MNIST data loaded")
ourdata_x = []
ourdata_y = []
for image_index in range(10):
    image = Image.open("images/" + str(image_index) + ".jpg").convert('L')
    array = np.array(image)
    array = 1 - np.resize(array, ([28, 28]))
    ourdata_x.append(array)
    ourdata_y.append(image_index)
x_ours = np.array(ourdata_x)
y_ours = np.array(ourdata_y)
print("Our data loaded")

# Hyperhyperparameters
ACTIVATORS = ["elu", "relu", "selu", "softmax", "softsign", "tanh", "hard_sigmoid", "sigmoid", "linear"]
OPTIMIZERS = ["sgd", "adamax", "adam", "nadam", "adadelta", "adagrad", "rmsprop"]
LOSSES = ["sparse_categorical_crossentropy"]

# Hyperparameters
start_time = time.time()
OPTIMIZER = "adam"
LEARNING_RATE = 0
LOSS = "categorical_hinge"
EPOCHS = 512
METRICS = ['accuracy']
STOCH_BATCH = 256
ROUNDING = 6

# Layer definition
LAYERS = "784-500-relu-dropout_0.2-10-softmax"
"""
Each layer is separated by a dash.
Pure number is a simple linear layer
Within each layer string, an underscore (_) separates function from parameters
i.e. 'dropout_0.2' is 20% dropout
"""

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
    
# Define model
print("Building model")
model = keras.models.Sequential()
layers_split = LAYERS.lower().split("-")
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
print("Model fitting parameters: optimizer={0}, loss={1}, lr={2}".format(OPTIMIZER, LOSS, LEARNING_RATE))
model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS)
print("Beginning fit")
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=STOCH_BATCH)
def evaluate_set(x_set, y_set):
    scores = model.evaluate(x_set, y_set)
    statistics = dict()
    for i in range(len(METRICS)):
        key = model.metrics_names[i + 1]
        value = scores[i + 1]
        statistics[key] = value
    return statistics
print("Training finished. Gathering data")

# Gather evaluation statistics
guess = model.predict(x_ours)
train_stats = evaluate_set(x_train, y_train)
train_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in train_stats.items()]
test_stats = evaluate_set(x_test, y_test)
test_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in test_stats.items()]
our_stats = evaluate_set(x_ours, y_ours)
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
