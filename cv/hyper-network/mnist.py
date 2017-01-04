from mxnet.initializer import Xavier

from facility import load_mnist
from mx_layers import *
from mx_solver import MXSolver

data = load_mnist(shape=(1, 28, 28))

network = variable('data')
network = convolution(network, (7, 7), 16, (1, 1), (3, 3))
network = ReLU(network)
network = pooling(network, 'maximum', (2, 2), (2, 2))
network = convolution(network, (7, 7), 16, (1, 1), (3, 3))
network = ReLU(network)
network = pooling(network, 'maximum', (2, 2), (2, 2))
network = flatten(network)
network = fully_connected(network, 10)
network = softmax_loss(network)

optimizer_settings = {
  'initial_lr' : 0.05,
  'optimizer'  : 'SGD',
}

solver = MXSolver(
  batch_size = 1000,
  data = data,
  devices = (0,),
  epochs = 5,
  symbol = network,
  initializer = Xavier(),
  optimizer_settings = optimizer_settings,
)

test_accuracy, progress = solver.train()
print test_accuracy, progress
