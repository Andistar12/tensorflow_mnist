# Tensorflow + MNIST

A project to read my own handwriting using machine learning. This program can train both regular neural networks and convoluted neural networks under the MNIST dataset WITHOUT writing code to make networks. All features:

* Config file to specify networks to train. No need to write code to make networks
* Train multiple networks at one time (in series)
* TensorBoard-friendly output (scalar, graph, and histogram only)
* Run your own digit images through the network and see if it's recognized
* Saves model weights, which can be recovered later
* Proper logging

## Training and viewing results

This project uses Python 3 and pipenv to manage libraries. Tensorflow has to be installed via `pipenv run pip3 install tensorflow`, but the other libraries can be installed simply with `pipenv install`. 

To train networks, edit the `config.json` file first to specify networks to train. Logging and output folder may also be specified from the config file. To run training, simply run `pipenv run python3 mnist.py`.

To view TensorBoard results, run `pipenv run tensorboard --logdir <training data location>` and open a browser to `localhost:6006`.

## Best results

Network: etc

Training data: xx.xx% accuracy

Validation data: xx.xx% accuracy

My handwriting: 100.00% accuracy

## Technical details

### Config file explanation

The config file has the following top-level parameters:

Parameter | Data Type | Value
--------- | --------- | -----
`log_prefix` | string | Prefix string for log files. If blank, no logs will be saved
`log_format` | string | Logging format for each log entry
`storage_loc` | string | Folder to store training data in
`epochs` | int | Number of epochs to train all networks for
`epoch_update` | int | How often to save a checkpoint for TensorBoard graphs
`batch` | int | For stochastic training, the size of each batch run through the network
`networks` | see below | A list of networks to train

#### Network structure

The config file can be used to create networks to train. Some parameters accept a list instead of just a single value. If a list is provided, multiple networks will be created, one for each value in the list. If multiple lists are present, all permutations of networks will be created. 

Networks have the following parameters:

Parameter | Data Type | Value
--------- | --------- | -----
`name` | string | Name 
`layers` | string | The one-liner that specifies the network structure
`optimizer` | string list | A list of optimizer to 
`learning_rate` | float list | The learning rate to use
`loss` | string | The loss function to use
`input_shape` | tuple | Recommended to keep this as `[28, 28]` for MNIST

The following optimizers are available: `sgd`, `adamax`, `adam`, `nadam`, `adadelta`, `adagrad`, `rmsprop`

#### The layer string

The layer string is the complete definition of a neural network. Each layer is separated by a dash (`-`), and parameters are separated by an underscore (`_`). The following layers are defined:

* Dense: No extra parameters. Example: `800` is a dense layer of 800 hidden neurons.
* Flatten: No extra parameters. Usage: `flatten`.
* Activations: No parameters. The following activations are available: `elu`, `relu`, `selu`, `softmax`, `softsign` `tanh`, `hard_sigmoid`, `sigmoid`, `linear`.
* Reshape: Each parameter is another dimension. Example: `reshape_28_28_1` reshapes the input to be of dimension (28, 28, 1).
* Embedded layer: parameters are vocab size and neurons. Example: `embed_2000_32` is an embedded layer with a vocab size of 2000 and 32 hidden neurons.
* Convoluted 2D: Parameters are feature maps, kernel size (square), and stride length. Example: `conv2d_32_5_1` represents a 2D convoluted layer with 32 features, a square kernel of dimensions 5x5, and a stride of 1.
* MaxPool: Parameters are pool size. Example: `maxpool_2` uses pools of size 2x2.
* L1/L2 regularization: Parameters are the weight. Example: `l2_0.01` represents an L2 regularization with weight 0.01.
* Dropout: Parameters are the probability. Example: `dropout_0.25` represents a dropout layer with a drop chance of 25%.

Example layer string: `flat-800-sigmoid-dropout_0.25-10-softmax` declares a flatten layer, a dense layer with 800 hidden nodes, a sigmoid activation, a dropout layer with probability 0.25, a dense layer with 10 hidden nodes, and a softmax activation.


### File breakdown

* `mnist.py`: The main script to train networks
* `network.py`: Framework/support file that defines the Network class
* `config.json`: The config file to edit networks in
* `Pipfile`, `Pipfile.lock`: Used to install libraries
* `old`: Older iterations of the project, before `mnist.py`
* `images/`: Individual images that get checked 
* `train_data`: Pre-trained networks that can be viewed in TensorBoard