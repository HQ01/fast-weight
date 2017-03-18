from argparse import ArgumentParser
import cPickle as pickle
import sys

from data_utilities import load_cifar10_record
from gpu_utility import gpu_availability
from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from decayed_attention_network import decayed_attention_network

parser = ArgumentParser()
parser.add_argument('--n_layers', type=int)
parser.add_argument('--decaying_rate', type=float)
parser.add_argument('--gpu', type=str)
configs = parser.parse_args()

settings = (
  {
    'operator' : 'convolution',
    'kwargs'   : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },
  {
    'operator' : 'decayed_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers'             : configs.n_layers,
      'weight_sharing'       : False,
      'memory_settings'      : {'decaying_rate' : configs.decaying_rate, 'learning_rate' : 1.0,},
    },
  },
  {'operator'  : 'transit', 'n_filters' : 32},
  {
    'operator' : 'decayed_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 32, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers'             : configs.n_layers,
      'weight_sharing'       : False,
      'memory_settings'      : {'decaying_rate' : configs.decaying_rate, 'learning_rate' : 1.0,},
    },
  },
  {'operator'  : 'transit', 'n_filters' : 64},
  {
    'operator' : 'decayed_attention_module',
    'settings' : {
      'convolution_settings' : {'n_filters' : 64, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
      'n_layers'             : configs.n_layers,
      'weight_sharing'       : False,
      'memory_settings'      : {'decaying_rate' : configs.decaying_rate, 'learning_rate' : 1.0,},
    },
  },
)

network = decayed_attention_network(settings)

BATCH_SIZE = 128
lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}
lr_scheduler = AtIterationScheduler(lr, lr_table)

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

DEVICES = map(int, configs.gpu.split(','))
solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = DEVICES,
  epochs = 1,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'cifar-decayed-memory-network-%d-decaying-rate-%f' % (configs.n_layers, configs.decaying_rate) 
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
