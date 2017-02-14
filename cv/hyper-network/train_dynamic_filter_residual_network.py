import cPickle as pickle
import numpy as np0
import sys
import mxnet as mx

from lr_scheduler import AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_initializer import HybridInitializer, PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

devices = (0,) # TODO
devices = (0, 1, 2, 3)
BATCH_SIZE = 128
DATA_SHAPE = (BATCH_SIZE / len(devices), 3, 32, 32)
MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'data_shape' : DATA_SHAPE}
# MODES = {'mode' : 'hyper', 'embedding' : 'parameter'}
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

data = load_cifar10_record(BATCH_SIZE)
lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : AtIterationScheduler(lr, lr_table),
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size          = BATCH_SIZE,
  devices             = devices,
  epochs              = 150,
  initializer         = PReLUInitializer(),
  optimizer_settings  = optimizer_settings,
  symbol              = network,
  verbose             = True,
)

info = solver.train(data)

identifier = 'dynamic-filter-triple-state-residual-network-%d' % N
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
