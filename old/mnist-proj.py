"""
Program to generate PCA visualizations with trained networks from mnist.py
This file is NOT compatible with TF 2.0 and should be dropped when porting to TF 2.0
Networks must be trained with mnist.py before being processed with this 
"""

from __future__ import print_function
import time
import datetime
import numpy as np
import logging
import sys
import json
import os
from PIL import Image
from network import Network 
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

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
PCA_BATCH = config.get("pca_batch", 1024)

print("Config loaded")

# Load in MNIST train and test data
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = keras.datasets.mnist.load_data()
X_TEST = X_TEST.astype("float32")
print("MNIST test data loaded")

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
Y_OURS = np.array(Y_OURS)

# Load in networks
networks = list()
networks_data = config.get("networks", list())
if len(networks_data) > 0:
    print("Found {0} network entries to scan in".format(len(networks_data)))
    for entry in networks_data:

        # Verify all data is valid
        name = entry.get("name", "")
        if name is "":
            print("Network has no name. Skipping")
            continue
        layers = entry.get("layers", "")
        if layers is "":
            print("Network has no layer data. Skipping")
            continue
        optimizer = entry.get("optimizer", "")
        if optimizer is "":
            print("Network has no optimizer. Skipping")
            continue
        learning_rate = entry.get("learning_rate", 0.0)
        loss = entry.get("loss", "")
        if loss is "":
            print("Network has no loss. Skipping")
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


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

# Gen sprite and tsv data for mnist
sprites_test_loc = "images/sprites-mnist.jpg"
sprites_ours_loc = "images/sprites-ours.jpg"
metadata_test_loc = "images/metadata_test.tsv"
metadata_ours_loc = "images/metadata_ours.tsv"
x_batch = 255 - X_TEST[0:PCA_BATCH]
sprite = create_sprite_image(x_batch)
result_mnist = Image.fromarray((sprite).astype(np.uint8))
sprite = create_sprite_image(X_OURS)
result_ours = Image.fromarray((sprite).astype(np.uint8))
y_batch = Y_TEST[0:PCA_BATCH]

print("Networks loaded. Beginning PCA gen of {0} networks".format(len(networks)))

for net in networks:
    # Load weights of network
    storage_loc = NETWORKS_LOC + net.name + "/"
    weights_loc = storage_loc + net.name + ".hdf5"
    if not os.path.isfile(weights_loc):
        # Weights don't exist. Network not trained
        print("No weights found for network {0}. Skipping".format(net.name))
        continue

    if not os.path.isdir(storage_loc):
        os.mkdir(storage_loc)
    if not os.path.isdir(storage_loc + "/images"):
        os.mkdir(storage_loc + "/images")

    # Write MNIST data
    result_mnist.save(storage_loc + sprites_test_loc)
    result_ours.save(storage_loc + sprites_ours_loc)
    with open(storage_loc + metadata_test_loc, "w") as metadata_file:
        for row in y_batch:
            metadata_file.write(str(row) + "\n")
    with open(storage_loc + metadata_ours_loc, "w") as metadata_file:
        for row in Y_OURS:
            metadata_file.write(str(row) + "\n")

    # Recover network
    print("Building network " + net.name)

    net.build()
    with tf.name_scope("proj"):
        net.load_weights(weights_loc)

        # Get layers to track
        layers = [net.layer_list[-4]]
        layers = [layer for layer in net.model.layers if layer.name in layers]
     
        # Configure TF and projector
        sess = tf.keras.backend.get_session()
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        test_functions = list()
        ours_functions = list()

        # Set embedding vars and attach to layer
        for layer in layers:

            # Create TF variables to monitor input data
            with tf.name_scope("proj_" + layer.name):
                embedding_test = tf.Variable(tf.zeros([PCA_BATCH, np.prod(layer.output_shape[1:])]), name=layer.name + "-mnist")
                embedding_ours = tf.Variable(tf.zeros([10, np.prod(layer.output_shape[1:])]), name=layer.name + "-ours")
                test_functions.append(embedding_test.assign(layer.output))
                ours_functions.append(embedding_ours.assign(layer.output))

                # Add to Projector
                proj_test = config.embeddings.add()
                proj_test.tensor_name = embedding_test.name
                proj_test.sprite.image_path = sprites_test_loc
                proj_test.metadata_path = metadata_test_loc
                proj_test.sprite.single_image_dim.extend([28, 28])
                proj_ours = config.embeddings.add()
                proj_ours.tensor_name = embedding_ours.name
                proj_ours.sprite.image_path = sprites_ours_loc
                proj_ours.metadata_path = metadata_ours_loc
                proj_ours.sprite.single_image_dim.extend([28, 28])

        # Setup writer
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(storage_loc)
        writer.add_graph(sess.graph)
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

        # Run functions and save
        x = net.model.input

        for function in test_functions:
            sess.run(function, feed_dict={x: x_batch})
        for function in ours_functions:
            sess.run(function, feed_dict={x: X_OURS})
        saver.save(sess, os.path.join(storage_loc, "{0}_{1}.ckpt".format(net.name, layer.name)))

    # Reset
    keras.backend.clear_session()
