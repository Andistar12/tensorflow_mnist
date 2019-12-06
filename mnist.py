# File to train MNIST networks from a config.json file of the same directory
# This script should be compatible with TF 2.0 

from __future__ import print_function
import time
import datetime
import numpy as np
import logging
import sys
import json
import os
from PIL import Image
from tensorflow import summary as tfs
from tensorflow import keras
from network import Network 
from tensorflow.keras import backend as K

# Load config file
try:
    cfgfile = open("./config.json")
    config = json.load(cfgfile)
except Exception:
    print("Error occurred opening config file config.json. Exiting")
    sys.exit(1)

# Get Hyperhyperparameters
ROUNDING = int(config.get("rounding", 6))
NETWORKS_LOC = config.get("storage_loc", "./networks")
if not NETWORKS_LOC.endswith("/"):
    NETWORKS_LOC += "/"
log_prefix = config.get("log_prefix", "")
log_format = config.get("log_format", "[%(levelname)s] [%(name)s %(asctime)s] %(message)s")
epochs = config.get("epochs", 0)
epoch_update = config.get("epoch_update", 0)
batch = config.get("batch", 0)

# Debug-level logging to file, info-level logging to console
logFormat = logging.Formatter(log_format)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

# Set up debug-level logging to file
if log_prefix is not None:
    currdatetime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
    log_file = "{0}_{1}.log".format(log_prefix, currdatetime)
    fileHandler = logging.FileHandler(filename=log_file, encoding='utf-8', mode='w')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormat)
    rootLogger.addHandler(fileHandler)

# Set up info-level logging to console
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormat)
rootLogger.addHandler(consoleHandler)

logger = logging.getLogger("main")
logger.debug("Logging setup for console and file")

# Load in MNIST train and test data
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()
X_TRAIN = X_TRAIN.astype("float32")
X_TEST = X_TEST.astype("float32")
logger.info("MNIST train and test data loaded")
Y_TRAIN = keras.utils.to_categorical(Y_TRAIN, 10)
Y_TEST = keras.utils.to_categorical(Y_TEST, 10)

# Load in our data
X_OURS = []
Y_OURS = []
for image_index in range(10):
    image = Image.open("images/" + str(image_index) + ".jpg").convert("L")
    array = np.array(image)
    array = np.resize(array, (28,28))
    X_OURS.append(array)
    Y_OURS.append(image_index)
X_OURS = np.array(X_OURS).astype("float32")
Y_OURS = keras.utils.to_categorical(Y_OURS, 10)

# Load in networks
networks = list()
networks_data = config.get("networks", list())
if len(networks_data) > 0:
    logger.debug("Found {0} network entries to scan in".format(len(networks_data)))
    for entry in networks_data:

        # Verify all data is valid
        name = entry.get("name", "")
        if name is "":
            logger.warn("Network has no name. Skipping")
            continue
        layers = entry.get("layers", "")
        if layers is "":
            logger.warn("Network has no layer data. Skipping")
            continue
        optimizer = entry.get("optimizer", "")
        if optimizer is "":
            logger.warn("Network has no optimizer. Skipping")
            continue
        learning_rate = entry.get("learning_rate", 0.0)
        loss = entry.get("loss", "")
        if loss is "":
            logger.warn("Network has no loss. Skipping")
            continue
        input_shape = tuple(entry.get("input_shape", (0))) # TODO cleanup
        metrics = entry.get("metrics", ["accuracy"])

        # Create lists of what can vary
        if isinstance(optimizer, str):
            optimizer = [optimizer]
        if isinstance(learning_rate, float):
            learning_rate = [learning_rate]
        if isinstance(loss, str):
            loss = [loss]

        for op in optimizer:
            for lr in learning_rate:
                for l in loss:
                    network_name = "{0}-op={1},lr={2},ls={3}".format(name, op, lr, l)
                    networks.append(Network(network_name, layers, op, lr, l, input_shape, metrics=metrics, log_name=name))

logger.info("Networks loaded. Beginning training of {0} networks".format(len(networks)))

"""
Each layer is separated by a dash.
Pure number is a simple linear layer
Within each layer string, an underscore (_) separates function from parameters
i.e. 'dropout_0.2' is 20% dropout
"""

for net in networks:
    # Build network and setup callbacks
    logger.info("Building network " + net.name)
    net.build()
    storage_loc = NETWORKS_LOC + net.name + "/"

    with K.name_scope("callbacks"):
        tb = keras.callbacks.TensorBoard(
                log_dir=storage_loc,
                histogram_freq=epoch_update,
                batch_size=batch,
                write_graph=True,
                write_grads=False,
                write_images=False,

                # Bugged, check here in the future
                # https://github.com/tensorflow/tensorboard/issues/2074
                # embeddings_freq=1,
                # embeddings_data=X_OURS,
                # embeddings_layer_names=None,
                # embeddings_metadata="/images/metadata.tsv",

                update_freq="epoch")
        checkpoint = keras.callbacks.ModelCheckpoint(
                storage_loc + net.name + ".hdf5",
                monitor="val_loss", 
                save_best_only=True, 
                save_weights_only=True,
                mode="auto",
                period=epoch_update)
        callbacks = [checkpoint, tb]

    logger.info("Training network " + net.name)
    history = net.train(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, epochs, batch, callbacks)

    if history == None:
        logger.warn("No history generated. Skipping")
        continue

    # Gather evaluation statistics
    logger.info("Gathering statistics")
    train_stats = net.evaluate(X_TRAIN, Y_TRAIN)
    train_stats = ["{0} {1}".format(k, round(float(v), ROUNDING)) for k, v in train_stats.items()]
    test_stats = net.evaluate(X_TEST, Y_TEST)
    test_stats = ["{0} {1}".format(k, round(float(v), ROUNDING)) for k, v in test_stats.items()]
    our_stats = net.evaluate(X_OURS, Y_OURS)
    our_stats = ["{0} {1}".format(k, round(float(v), ROUNDING)) for k, v in our_stats.items()]
    guess = net.guess(X_OURS)
    guess_argmax = [np.argmax(a) for a in guess]

    # Output
    logger.info("----------------------- TEST RESULTS ------------------------")
    logger.info("Network name: " + net.name)
    logger.info("Train stats: " + " ".join(train_stats))
    logger.info("Test stats: " + " ".join(test_stats))
    logger.info("Our stats: " + " ".join(our_stats))
    logger.info("Guesses: " + str(guess_argmax))
    logger.info("-------------------------------------------------------------")

    # Reset
    keras.backend.clear_session()

    # Write guesses to log
    writer = tfs.create_file_writer("ours")
    with writer.as_default():
        for i in range(10):
            for j in range(10):
                tfs.scalar("our_data_" + str(i), guess[i][j], step=j)
                writer.flush()

logger.info("All training finished. Quitting")
