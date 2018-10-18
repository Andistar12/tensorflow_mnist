import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR) #Console logging level
LOSS_DECIMALS = 6
ACC_DECIMALS = 3
SOFTMAX_DECIMALS = 3

#Network hyper-parameters
LEARNING_RATE = 1.7
EPOCHS = 25000
EPOCH_UPDATE = 1000
NETWORK = [784, 50, 30, 12, 10]
REGULARIZATION = 0.0001
MINIBATCH_TRAIN = 125
MINIBATCH_CROSSVALID = 1000

#Network parameters
weights = []
biases = []

#Train and test data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels

ourdata_x = []
ourdata_y = []
for i in range(10):
	image = Image.open("images/" + str(i) + ".jpg").convert('L')
	array = np.array(image)
	array = 1 - np.resize(array, ([784]))
	ourdata_x.append(array)
		
	onehot = np.zeros((10))
	onehot[i] = 1
	ourdata_y.append(onehot)
	
	#Show image
	#image = Image.fromarray(np.resize(array, [28, 28]))
	#image.show()


def softmax_string(softmax):
	result = []
	for element in softmax:
		result.append(str(round(element, SOFTMAX_DECIMALS)))
	return "[" + ", ".join(result) + "]"

def neural_network(x_input_layer, layer_neurons):
	
	"""
	Layer neurons contains the neurons of every layer and must have a min length = 2
	weights use xavier's init (truncated to 2 stddev)
	bias uses normal curve init
	all layers use sigmoid, last uses softmax
	"""
	
	curr_layer_eval = x_input_layer
	curr_layer = -1
	
	if len(layer_neurons) > 2:
		for curr_layer in range(len(layer_neurons) - 2):
			curr_layer_neurons = layer_neurons[curr_layer]
			next_layer_neurons = layer_neurons[curr_layer + 1]
			w = tf.get_variable("weight" + str(curr_layer), shape=[curr_layer_neurons, next_layer_neurons], initializer=tf.random_normal_initializer())
			b = tf.get_variable("bias" + str(curr_layer), shape=[next_layer_neurons], initializer=tf.random_normal_initializer())
			weights.append(w)
			biases.append(b)
			curr_layer_eval = tf.nn.sigmoid(tf.matmul(curr_layer_eval, w) + b)
			
	curr_layer += 1
	curr_layer_neurons = layer_neurons[curr_layer]
	next_layer_neurons = layer_neurons[curr_layer + 1]
	w = tf.get_variable("weight" + str(curr_layer), shape=[curr_layer_neurons, next_layer_neurons], initializer=tf.random_normal_initializer())
	b = tf.get_variable("bias" + str(curr_layer), shape=[next_layer_neurons], initializer=tf.random_normal_initializer())
	weights.append(w)
	biases.append(b)
	return tf.nn.softmax(tf.matmul(curr_layer_eval, w) + b)

#Var initiation - use the neural network above
x = tf.placeholder(tf.float32, [None, 784], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")
yhat = neural_network(x, NETWORK)

#TF Functions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
if REGULARIZATION > 0:
	regularizer = tf.reduce_sum( [tf.nn.l2_loss(x) for x in weights] )
	loss = tf.reduce_mean(loss + REGULARIZATION * regularizer)
train_model = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#Compare if same index of highest prediction, cast to float, average
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat,1), tf.argmax(y,1)), tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	current_loss = 0
	current_acc = 0
	current_epoch = 0
	while current_epoch < EPOCHS:
		current_epoch = current_epoch + 1
		train_x, train_y = mnist.train.next_batch(MINIBATCH_TRAIN)
		sess.run(train_model, feed_dict={x: train_x, y: train_y})

		if current_epoch % EPOCH_UPDATE == 0:
			check_x, check_y = mnist.train.next_batch(MINIBATCH_CROSSVALID)
			current_loss = sess.run(loss, feed_dict={x: train_x, y: train_y})
			current_acc = 100 * sess.run(accuracy, feed_dict={x: check_x, y: check_y})
			print("Epoch {0}: curr_loss={1}, curr_acc={2}% ".format( format(current_epoch, "06"), format(current_loss, ".{0}f".format(LOSS_DECIMALS)), format(current_acc, ".{0}f".format(ACC_DECIMALS))))
	
	mnist_accuracy = 100 * sess.run(accuracy, feed_dict={x: test_x, y: test_y})
	
	#our_accuracy = 100 * sess.run(accuracy, feed_dict={x: ourdata_x, y: ourdata_y})
	#print("Final self accuracy: " + str(our_accuracy))
	
	self_acc = 0
	for i in range(10):
		softmax = sess.run(yhat, feed_dict={x: [ourdata_x[i]]})[0]
		guess = np.argmax(softmax)
		if guess - i == 0:
			self_acc += 1
		print("{0}: {1}, softmax {2}".format(i, guess, softmax_string(softmax)))
	print("Self accuracy: {0}/10".format(self_acc))
	print("Final database accuracy: " + str(mnist_accuracy))
