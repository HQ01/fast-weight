import cPickle as pickle
import sys

import mxnet as mx
from mxnet.initializer import Xavier
from lr_scheduler import FactorScheduler
from facility import load_mnist
from mx_layers import *
from mx_solver import MXSolver

try: mode = sys.argv[1]
except: mode = 'hyper'

network = variable('data')
network = convolution(network, (7, 7), 16, (1, 1), (3, 3))
network = ReLU(network)
network = pooling(network, 'maximum', (2, 2), (2, 2))

if mode == 'normal':
  weight = variable('convolution1_weight', shape=())
else:
  # weight shape (FILTER_OUT, FILTER_IN, WIDTH, HEIGHT)
  N_z = 4
  d = N_z
  FILTER_IN = 16
  FILTER_OUT = 16
  WIDTH = 7
  HEIGHT = 7
  weight = variable('embedding', shape=(1, N_z))
  weight = fully_connected(weight, FILTER_IN * d)
  weight = reshape(weight, (FILTER_IN, d))
  weight = fully_connected(weight, FILTER_OUT * WIDTH * HEIGHT)
  weight = reshape(weight, (FILTER_IN, FILTER_OUT, WIDTH, HEIGHT))
  weight = swap_axes(weight, 0, 1)

network = convolution(network, (7, 7), 16, (1, 1), (3, 3), weight=weight)
# network = convolution(network, (7, 7), 16, (1, 1), (3, 3))
network = ReLU(network)
network = pooling(network, 'maximum', (2, 2), (2, 2))
network = flatten(network)
network = fully_connected(network, 10)
network = softmax_loss(network)

class Initializer(Xavier):
  def _init_default(self, arg, array):
    if arg == 'embedding':
      mx.random.normal(0, 0.01, out=array)

data = load_mnist(shape=(1, 28, 28))

BATCH_SIZE = 1000
lr = 0.005

optimizer_settings = {
  'initial_lr' : lr,
  'optimizer'  : 'Adam',
  'lr_scheduler' : FactorScheduler(lr, 0.99, data[0].shape[0] // BATCH_SIZE),
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0,),
  epochs = 50,
  initializer = Initializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = False,
)

identifier = 'mnist-%s-network' % mode
info = solver.train(data)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
