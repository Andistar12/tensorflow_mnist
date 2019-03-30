from __future__ import print_function
import time
import datetime
import numpy as np
import logging
import sys
import json
import os
from tensorflow import keras
from PIL import Image
from network import Network

# TODO add logging params in config file

# TODO move to config file?
ROUNDING = 6
NETWORKS_LOC = "./networks"

# Debug-level logging to file, info-level logging to console
logFormat = logging.Formatter('[%(levelname)s] [%(name)s %(asctime)s] %(message)s')
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

# Set up logging to file
currdatetime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
log_file = "log_{0}.log".format(currdatetime)
fileHandler = logging.FileHandler(filename=log_file, encoding='utf-8', mode='w')
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormat)
rootLogger.addHandler(fileHandler)

# Set up logging to console
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormat)
rootLogger.addHandler(consoleHandler)

logger = logging.getLogger("main")
logger.debug("Logging setup for console and file")

# Load in MNIST data
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()
X_TRAIN = X_TRAIN.astype("float32") / 255.0
X_TEST = X_TEST.astype("float32") / 255.0
logger.info("MNIST train and test data loaded")

# Load in our data
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
logger.info("Personal data loaded")

# Load in networks
# TODO account for metrics
def gen_network(file):
    with open(file) as json_file:
        logger.info("Reading in network from JSON file " + file)
        data = json.load(json_file)
        if data.get("train_data", "") is not "":
            logger.warn("Network already traineded. Skipping")
            return None
        name = data.get("name", "")
        if name is "":
            logger.warn("Network has no name. Skipping")
            return None
        layers = data.get("layers", "")
        if layers is "":
            logger.warn("Network has no layer data. Skipping")
            return None
        optimizer = data.get("optimizer", "")
        if optimizer is "":
            logger.warn("Network has no optimizer. Skipping")
            return None
        learning_rate = data.get("learning_rate", 0.0)
        loss = data.get("loss", "")
        if loss is "":
            logger.warn("Network has no loss. Skipping")
            return None
        input_shape = tuple(data.get("input_shape", (0)))

        return Network(name, layers, optimizer, learning_rate, loss, input_shape)
        # def __init__(self, name, layers, optimizer, learning_rate, loss, input_shape, metrics = ["accuracy"]):

# Create test networks
# network1 = Network("network1", "flat-100-relu-10-softmax", "adam", 0, "sparse_categorical_crossentropy", (28,28,))
#network2 = Network("network2", "flat-100-relu-10-softmax", "sgd", 0, "sparse_categorical_crossentropy", (28,28,))
# network3 = Network("network3", "reshape_28_28_1-conv2d_32_5_1-maxpool_2-flatten-512-relu-10-softmax", "adam", 0, "sparse_categorical_crossentropy", (28,28,))
# networks = [network3]

logger.debug("Reading in networks from directory " + NETWORKS_LOC)
networks = list()
for file in os.listdir(NETWORKS_LOC):
    if file.endswith(".json"):
        logger.debug("Found file to load: " + NETWORKS_LOC + file)
        network = gen_network(os.path.join(NETWORKS_LOC, file))
        if network is not None:
            networks.append(network)

logger.info("Networks loaded. Beginning training")

"""
Each layer is separated by a dash.
Pure number is a simple linear layer
Within each layer string, an underscore (_) separates function from parameters
i.e. 'dropout_0.2' is 20% dropout
"""

for net in networks:
    # Perform training and get loss
    logger.info("Building network " + net.name)
    net.build()
    logger.info("Training network " + net.name)
    history = net.train(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, 5, 125)

    if history == None:
        logger.warn("No history generated. Skipping")
        continue

    # Gather evaluation statistics
    logger.info("Gathering statistics")
    guess = net.guess(X_OURS)
    train_stats = net.evaluate(X_TRAIN, Y_TRAIN)
    train_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in train_stats.items()]
    test_stats = net.evaluate(X_TEST, Y_TEST)
    test_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in test_stats.items()]
    our_stats = net.evaluate(X_OURS, Y_OURS)
    our_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in our_stats.items()]

    # Output
    logger.info("----------------------- TEST RESULTS ------------------------")
    logger.info("Train stats: " + " ".join(train_stats))
    logger.info("Test stats: " + " ".join(test_stats))
    logger.info("Our stats: " + " ".join(our_stats))
    logger.info("Our predictions: " + str([np.argmax(x) for x in guess]))

    # Write train and weight data to file
    logger.info("Writing train data to file for " + net.name)
    filename = os.path.join(NETWORKS_LOC, net.name + ".json")
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    data["train_data"] = history
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info("Writing train data weights to file for " + net.name)
    filename = os.path.join(NETWORKS_LOC, net.name + ".h5")
    net.save_weights(filename)
    logger.info("Wrote train data to file for network " + net.name)
logger.info("All training finished. Quitting")
