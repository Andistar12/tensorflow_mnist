import time
import numpy as np
import logging
import json
from tensorflow import keras

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

# The general network class
class Network:

    def __init__(self, name, layers, optimizer, learning_rate, loss, input_shape, metrics = ["accuracy"], log_name=None):
        # Save vars
        if log_name is None:
            log_name = name
        self.logger = logging.getLogger(log_name)
        self.name = name # String of given name of network
        self.layers = layers # String of custom layer definition
        self.optimizer = optimizer # String of function name
        self.loss = loss # String of fnction name
        self.learning_rate = learning_rate # Float of learning rate
        self.metrics = metrics # Array of strings of tf.metrics
        self.input_shape = input_shape # Tuple of shape of input
        self.model = None
    
    def build(self):
        logger = self.logger
        logger.info("Creating model " + self.name)
        try:
            inputs = keras.layers.Input(shape=self.input_shape)
            prev_layer = inputs
            layers_split = self.layers.lower().split("-")
            for layer in layers_split:
                
                # Raw numberical is dense layer
                if is_int(layer):
                    neurons = int(layer)
                    if neurons < 0:
                        logger.warn("Dense layer recognized but neurons is < 0. Skipping")
                        continue
                    logger.debug("Dense layer recognized with hidden neurons=" + layer)
                    prev_layer = keras.layers.Dense(neurons)(prev_layer)
                
                # General activation functions
                elif layer in ACTIVATORS:
                    logger.debug("Activation function recognized with function=" + layer)
                    prev_layer = keras.layers.Activation(activation=layer)(prev_layer)

                # Flatten layer
                elif "flat" in layer:
                    logger.debug("Flatten function recognized")
                    prev_layer = keras.layers.Flatten()(prev_layer)

                elif "reshape" in layer:
                    conv_split = layer.split("_")
                    shape = tuple([int(x) for x in conv_split[1:]])
                    logger.debug("Reshape layer recognized with shape=" + str(shape))
                    prev_layer = keras.layers.Reshape(shape)(prev_layer)

                # Convoluted layer
                elif "conv2d" in layer:
                    conv_split = layer.split("_")
                    if len(conv_split) < 4 or not is_int(conv_split[1]) or not is_int(conv_split[2]) or not is_int(conv_split[3]):
                        logger.warn("Conv2D layer recognized but invalid params; skipping")
                        continue
                    feature_maps = int(conv_split[1])
                    kernel_size = int(conv_split[2])
                    stride_length = int(conv_split[3])
                    if feature_maps < 1 or kernel_size < 1 or stride_length < 1:
                        logger.warn("Conv2D layer recognized but params are < 1; skipping")
                        continue
                    logger.debug("Conv3D layer recognized with features={0} kernel=({1},{1}) strides=({2},{2})".format(feature_maps, kernel_size, stride_length))
                    prev_layer = keras.layers.Conv2D(filters=feature_maps, kernel_size=kernel_size, strides=stride_length)(prev_layer) 

                # Max pooling 2d
                elif "maxpool" in layer:
                    maxpool_split = layer.split("_")
                    if len(maxpool_split) < 2 or not is_int(maxpool_split[1]):
                        logger.warn("MaxPool layer recognized but invalid rate; skipping")
                        continue
                    pool = int(maxpool_split[1])
                    if pool < 1:
                        logger.warn("MaxPool layer recognized but rate is < 1; skipping")
                        continue
                    logger.debug(("MaxPool layer recognized with rate=" + maxpool_split[1]))
                    prev_layer = keras.layers.MaxPool2D(pool)(prev_layer)

                # Dropout layer
                elif "dropout" in layer:
                    dropout_split = layer.split("_")
                    if len(dropout_split) < 2 or not is_float(dropout_split[1]):
                        logger.warn("Dropout layer recognized but invalid rate; skipping")
                        continue
                    dropout = float(dropout_split[1])
                    if not 0 < dropout < 1:
                        logger.warn("Dropout layer recognized but rate is not in range (0,1); skipping")
                        continue
                    logger.debug(("Dropout layer recognized with rate=" + dropout_split[1]))
                    prev_layer = keras.layers.Dropout(dropout)(prev_layer)

                # L1 regularization
                elif "l1" in layer:
                    l1_split = layer.split("_")
                    if len(l1_split) < 2 or not is_float(l1_split[1]):
                        logger.warn("L1 reg layer recognized but invalid factor; skipping")
                        continue
                    logger.debug("L1 reg layer reocognized with factor=" + l1_split[1])
                    prev_layer = keras.layers.ActivityRegularization(l1=float(l1_split))(prev_layer)

                # L2 regularization
                elif "l2" in layer:
                    l2_split = layer.split("_")
                    if len(l2_split) < 2 or not is_float(l2_split[1]):
                        logger.warn("L2 reg layer recognized but invalid factor; skipping")
                        continue
                    logger.debug("L2 reg layer reocognized with factor=" + l2_split[1])
                    prev_layer = keras.layers.ActivityRegularization(l2=float(l2_split))(prev_layer)

                else:
                    logger.warn("Unknown layer type: " + layer + ". Skipping")
            
            logger.debug("Model parameters: optimizer={0}, loss={1}, lr={2}".format(self.optimizer, self.loss, self.learning_rate))
            self.model = keras.models.Model(inputs=inputs, outputs=prev_layer)
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            logger.info("Model " + self.name + " finished creation")
        except Exception as e:
            logger.error("Error creating network: " + str(e))
            self.model = None
    
    def load_weights(self, file):
        if self.model is None:
            self.logger.debug("No model to load weights into")
            return
        self.model.load_weights(file)
        self.logger.info("Model weights loaded from file " + file)

    def train(self, train_x, train_y, test_x, test_y, epochs, stoch_batch, epoch_update=5, storage_loc=None):
        if self.model == None:
            return None
        
        callbacks = list()
        if storage_loc is not None:
            file_loc = storage_loc
            if not file_loc.endswith("/"):
                file_loc += "/"
            checkpoint = keras.callbacks.ModelCheckpoint(
                    file_loc + self.name + ".h5", 
                    monitor="val_loss", 
                    save_best_only=True, 
                    save_weights_only=True,
                    mode="auto",
                    period=epoch_update)
            tb = keras.callbacks.TensorBoard(
                    log_dir=storage_loc,
                    histogram_freq=epoch_update,
                    update_freq=epoch_update)
            callbacks.append(checkpoint)
            callbacks.append(tb)

            # keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

        self.logger.info("Beginning training on model " + self.name)
        train_time = time.time()

        history = self.model.fit(train_x, train_y, 
            epochs=epochs, batch_size=stoch_batch,
            validation_data=(test_x, test_y),
            callbacks=callbacks)

        train_time = time.time() - train_time # Seconds
        self.logger.info("Finished training model in " + str(train_time) + " seconds")
        return history.history 

    def evaluate(self, set_x, set_y):
        if self.model == None:
            return None

        scores = self.model.evaluate(set_x, set_y)
        
        # scores[0] is always loss, scores[1+] is other stats
        statistics = dict(zip(self.model.metrics_names, scores))
        return statistics

    def guess(self, set_x):
        if self.model == None:
            return None
        return self.model.predict(set_x)
