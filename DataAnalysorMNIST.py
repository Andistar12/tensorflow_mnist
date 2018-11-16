import tensorflow as tf
import numpy as np
import csv
from tensorflow.examples.tutorials.mnist import input_data

#Network hyper-parameters
tf.logging.set_verbosity(tf.logging.ERROR) #Console logging level
DECIMALS = 6 #Loss and accuracy display decimal rounding
LEARNING_RATE = 1.3
EPOCH_BATCH = 500 #Indefinite training will run forever, but after every EPOCH_BATCH we display some info
HIDDEN_LAYER1_NEURONS = 30 #Hidden weights/biases for the hidden layer 1
HIDDEN_LAYER2_NEURONS = 12 #Hidden weights/biases for the hidden layer 2

#Image pipeline parameters 
## TODO Currenly using MNIST data - change to pipe in our data
IMAGE_SIZE = 28 * 28 #784
OUTPUT_SIZE = 10 #1-hot vector of 10 possible digits
MNIST_BATCH_TRAIN_SIZE = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels
train_x, train_y = mnist.train.next_batch(MNIST_BATCH_TRAIN_SIZE)

#Creates a variable
def init_variable(name, shape):
	return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()) #Xavier's for non-zero
	
#Define model estimator
def neuralnetwork(x_input, input_size, num_hidden, num_hidden_2, output_size):
	#x_input is our layer 1 - the input layer
	
	"""
	TODO turn two hidden layers into x hidden layers defined by a list of neuron count per layer
	i.e. [784, 5, 10] would create a 3 layer neural network with 784 inputs, 5 hidden neurons, and 10 outputs
	"""
	
	#4 layer neural network
	
	#This is layer 2 - our first hidden layer of weights and biases, sigmoid activation
	w1 = init_variable("weight1", [input_size, num_hidden])
	b1 = init_variable("bias1", [num_hidden])
	layer2 = tf.nn.sigmoid(tf.matmul(x_input, w1) + b1)

	#This is layer 3 - our second hidden layer of weights and biases, sigmoid activation	
	w2 = init_variable("weight2", [num_hidden, num_hidden_2])
	b2 = init_variable("bias2", [num_hidden_2])
	layer3 = tf.nn.sigmoid(tf.matmul(layer2, w2) + b2)
	
	#This is our layer 4 - the output layer, softmax activation
	w3 = init_variable("weight3", [num_hidden_2, output_size])
	b3 = init_variable("bias3", [output_size])
	return tf.nn.softmax(tf.matmul(layer3, w3) + b3) 
	
	"""
	2 layer neural network
	
	#This is our layer 2 - the output layer, softmax activation
	w1 = init_variable("weight1", [input_size, output_size])
	b1 = init_variable("bias1", [output_size])
	return tf.nn.softmax(tf.matmul(x_input, w1) + b1) 
	"""

#Model initiation - use the neural network above
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="x")
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="y")
yhat = neuralnetwork(x, IMAGE_SIZE, HIDDEN_LAYER1_NEURONS, HIDDEN_LAYER2_NEURONS, OUTPUT_SIZE)

#Cross entropy loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

#Optimizer and actual model to train
train_model = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

#Accuracy checker - giant vector of True/False converted to 1s and 0s, which is then averaged. 1 = good, 0 = bad
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat,1), tf.argmax(y,1)), tf.float32))

#Session initialization
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	# Train loop
	current_loss = 0
	current_accuracy = 0
	highest_accuracy = 0
	current_epoch = 0
	while True:
		current_epoch = current_epoch + 1
		
		#Perform train
		sess.run(train_model, feed_dict={x: train_x, y: train_y})
		train_x, train_y = mnist.train.next_batch(MNIST_BATCH_TRAIN_SIZE) #MNIST stochaic data

		#Display info - current loss and accuracy
		if current_epoch % EPOCH_BATCH == 0:
			current_loss = format(sess.run([loss], feed_dict={x: train_x, y: train_y})[0], ".{0}f".format(DECIMALS))
			current_accuracy = 100 * sess.run(accuracy, feed_dict={x: test_x, y: test_y})
			if current_accuracy > highest_accuracy:
				highest_accuracy = current_accuracy
			print("Epoch {0}: Loss={1}, curr_acc={2}% highest_acc={3}%".format( format(current_epoch, "07"), current_loss, format(current_accuracy, ".{0}f".format(DECIMALS)), format(highest_accuracy, ".{0}f".format(DECIMALS))))

