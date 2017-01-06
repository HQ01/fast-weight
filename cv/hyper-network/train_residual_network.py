import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

BATCH_SIZE = 128
MODES = {'mode' : 'normal'}
# MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'batch_size' : BATCH_SIZE}
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

lr = 0.01
# lr_table = {15 : 0.001, 30 : 0.0001}
lr_table = {32000 : 0.005, 48000 : 0.001}

data = load_cifar10_record(BATCH_SIZE)
n_training_samples = data[0].shape[0] if isinstance(data, np0.ndarray) else 50000 # TODO

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
# 'lr_scheduler' : AtEpochScheduler(lr, lr_table, n_training_samples, BATCH_SIZE), # TODO
  'lr_scheduler' : AtIterationScheduler(lr, lr_table),
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 1,
  initializer = Xavier(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

identifier = 'triple-state-%s-residual-network' % MODES['mode'] # TODO
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
