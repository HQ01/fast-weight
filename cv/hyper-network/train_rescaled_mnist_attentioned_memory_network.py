import cPickle as pickle
import gzip
import sys

from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from attended_memory_network import attended_memory_network
N = int(sys.argv[1])
# TODO change kernel size
settings = (
  {
    'operator' : 'convolution',
    'kwargs' : {'n_filters' : 8, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },
  {
    'operator' : 'pooling',
    'kwargs' : {'mode' : 'maximum', 'kernel_shape' : (2, 2), 'stride' : (2, 2), 'pad' : (0, 0)},
  },
  {
    'operator' : 'attended_memory_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 8, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers' : N,
      'probability' : 'softmax',
      'weight_sharing' : True,
    },
  }
)

network = attended_memory_network(settings)

BATCH_SIZE = 128
lr = 0.1
lr_table = {5000 : lr * 0.1, 7500 : lr * 0.01}
lr_scheduler = AtIterationScheduler(lr, lr_table)

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 150,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = pickle.load(gzip.open('rescaled_mnist/mnist.gz', 'rb'))
info = solver.train(data)

identifier = 'rescaled-mnist-attentioned-memory-network-%d' % N
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
