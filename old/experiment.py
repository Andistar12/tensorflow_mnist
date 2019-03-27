# Third neural network program, output standardization and abstraction

import tensorflow as tf
import numpy as np
from PIL import Image
import time
mnist = tf.keras.datasets.mnist


# Hyperparameters
start_time = time.time()
metrics=['accuracy']
EPOCHS = 5
STOCH_BATCH = 256
ROUNDING = 6


# Define model and trainer
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(500, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adamax',
              loss='sparse_categorical_crossentropy',
              metrics=metrics)


# Prepare datasets
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
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


# Train and define custom eval method
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=STOCH_BATCH)
def evaluate_set(x_set, y_set):
    scores = model.evaluate(x_set, y_set)
    statistics = dict()
    for i in range(len(metrics)):
        key = model.metrics_names[i + 1]
        value = scores[i + 1] 
        statistics[key] = value
    return statistics
print("Training finished. Gathering statistics")


# Gather evaluation statistics
guess = model.predict(x_ours)
train_stats = evaluate_set(x_train, y_train)
test_stats = evaluate_set(x_test, y_test)
our_stats = evaluate_set(x_ours, y_ours)
loss = history.history["loss"][-1]


# Output
print("----------------------- TEST RESULTS ------------------------\n")
print("Delta time (s): " + str(time.time() - start_time))
print("Final training loss: " + str(round(loss, ROUNDING)))
print()

print("Train stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in train_stats.items()]))
print("Test stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in test_stats.items()]))
print("Our stats: " + " ".join(["{0} {1}".format(key, round(value, ROUNDING)) for key, value in our_stats.items()]))
print()

print("Our predictions: " + str([np.argmax(x) for x in guess]))
print("Full softmax:\n" + str(guess))
