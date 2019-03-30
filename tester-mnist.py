from __future__ import print_function
import numpy as np
from PIL import Image
from tensorflow import keras
import sys # TODO REMOVE
import os
import json
from network import Network, gen_network

NETWORKS_LOC = "./networks"
ROUNDING = 6

# Load in MNIST data
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()
X_TRAIN = X_TRAIN.astype("float32") / 255.0
X_TEST = X_TEST.astype("float32") / 255.0
print("MNIST train and test data loaded")

# Load in our data
X_OURS = []
Y_OURS = []
for image_index in range(10):
    image = Image.open("images/" + str(image_index) + ".jpg").convert("L")
    array = np.array(image)
    array = 1 - np.resize(array, ([28,28]))
    X_OURS.append(array)
    Y_OURS.append(image_index)
X_OURS = np.array(X_OURS)
Y_OURS = np.array(Y_OURS)
print("Our data loaded")

# Read in availalbe networks by union of h5 and json files
print("Reading in networks from directory " + NETWORKS_LOC)
json_networks = set()
h5_networks = set()
for file in os.listdir(NETWORKS_LOC):
    if file.endswith(".json"):
        json_networks.add("".join(os.path.basename(file).split(".")[:-1]))
    elif file.endswith(".h5"):
        h5_networks.add("".join(os.path.basename(file).split(".")[:-1]))

# Union the two, display networks
networks = list(json_networks.intersection(h5_networks))

print("Found {0} items in directory {1}".format(len(networks), NETWORKS_LOC))
print(" ".join(networks))
net = input("Enter name to evaluate: ")

if net in networks:
    network = gen_network(os.path.join(NETWORKS_LOC, net + ".json"))
    if network is None:
        print("Error building network")
        sys.exit(1)
    network.build()
    network.load_weights(os.path.join(NETWORKS_LOC, net + ".h5"))

    # Gather evaluation statistics
    print("Gathering statistics")
    guess = network.guess(X_OURS)
    train_stats = network.evaluate(X_TRAIN, Y_TRAIN)
    train_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in train_stats.items()]
    test_stats = network.evaluate(X_TEST, Y_TEST)
    test_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in test_stats.items()]
    our_stats = network.evaluate(X_OURS, Y_OURS)
    our_stats = ["{0} {1}".format(k, round(v, ROUNDING)) for k, v in our_stats.items()]

    # Output
    print("\n----------------------- TEST RESULTS ------------------------\n")
    print("Train stats: " + " ".join(train_stats))
    print("Test stats: " + " ".join(test_stats))
    print("Our stats: " + " ".join(our_stats))
    print("Our predictions: " + str([np.argmax(x) for x in guess]))

else:
   print("No network found for given input. Quitting")
