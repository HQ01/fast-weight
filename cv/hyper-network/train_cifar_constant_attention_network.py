import cPickle as pickle
import sys

from data_utilities import load_cifar10_record
from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from constant_attention_network import constant_attention_network
N = int(sys.argv[1])
settings = (
  {
    'operator' : 'convolution',
    'kwargs'   : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },
  {
    'operator' : 'constant_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers' : N,
      'weight_sharing' : False,
    },
  },
  {
    'operator'  : 'transit',
    'n_filters' : 32,
  },
  {
    'operator' : 'constant_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 32, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers' : N,
      'weight_sharing' : False,
    },
  },
  {
    'operator'  : 'transit',
    'n_filters' : 64,
  },
  {
    'operator' : 'constant_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 64, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers' : N,
      'weight_sharing' : False,
    },
  },
)

network = constant_attention_network(settings)

BATCH_SIZE = 128
lr = 0.1
lr_table = {24000 : lr * 0.1, 48000 : lr * 0.01}
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

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'cifar-constant-memory-network-%d' % N
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
