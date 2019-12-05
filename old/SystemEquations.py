"""
Tensorflow program that uses gradient descent to solve systems of equations
"""

import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR) #Console logging level

#Network hyper-parameters
LEARNING_RATE = 0.003
EPOCHS = 10000
EPOCH_UPDATE = 1000
ROUNDING = 10

"""
4x +  y = 8
5x + 2y = 13
Solution: x=1, y=4

2x + 2y = 6
-x + 2y = 12
Solution: x=-2, y=5

 2x + 2y + z = -20
-3x +  y – 3z = 14
 3x + 3y – 3z = 6
Solution: x=1, y=-7, z=-8
"""

# Base functions
def func1(x, y):
    return 4 * x + 1 * y - 8
def func2(x, y):
    return 5 * x + 2 * y - 13

#Variables to calculate
x = tf.get_variable("x", shape=[1], initializer=tf.zeros_initializer())
y = tf.get_variable("y", shape=[1], initializer=tf.zeros_initializer())

#Functions to run
#cost = np.absolute(func1(x, y, z)) + np.absolute(func2(x, y, z)) + np.absolute(func3(x, y, z)) #MAE
cost = func1(x, y) ** 2 + func2(x, y) ** 2 #MSE
train_model = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    curr_cost = 0
    curr_epoch = 0
    
    while curr_epoch < EPOCHS:
        #Train
        curr_epoch = curr_epoch + 1
        sess.run(train_model)

        #Epoch update
        if curr_epoch % EPOCH_UPDATE == 0:
            curr_cost = sess.run(cost)
            print("Epoch {0}: cost={1}".format(curr_epoch, str(np.round(curr_cost, decimals=ROUNDING))))
    
    #Result
    curr_x, curr_y = np.round(sess.run([x, y]), decimals=ROUNDING)
    final_cost = sess.run(cost)
    print("\nExpecting x = 1, y = 4")
    print("Final result: x={0}, y={1}".format(str(curr_x[0]), str(curr_y[0])))
    val1 = str(np.round(func1(curr_x, curr_y), decimals=ROUNDING)[0])
    val2 = str(np.round(func2(curr_x, curr_y), decimals=ROUNDING)[0])
    print("Equation check (expt 0): func1={0}, func2={1}".format(val1, val2))
    print("Final cost: " + str(final_cost))
