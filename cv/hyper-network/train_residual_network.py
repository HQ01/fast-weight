import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

BATCH_SIZE = 128
MODES = {'mode' : 'normal'}
# MODES = {'mode' : 'weight-sharing'}
# MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'batch_size' : BATCH_SIZE}
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

lr = 0.01
lr_table = {32000 : 0.005, 48000 : 0.001}

data = load_cifar10_record(BATCH_SIZE)
n_training_samples = data[0].shape[0] if isinstance(data, np0.ndarray) else 50000 # TODO

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : AtIterationScheduler(lr, lr_table),
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 120,
  initializer = Xavier(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

# TODO
identifier = 'triple-state-%s-residual-network-%d' % (MODES['mode'], N) 
# identifier = 'triple-state-%s-residual-network-%d-%s-embedding' % (MODES['mode'], N, MODES['embedding'])
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
