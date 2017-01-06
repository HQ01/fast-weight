import cPickle as pickle
import numpy as np0

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

N = 3
# MODE = 'normal'
MODE = 'hyper'
network = triple_state_residual_network(N, MODE)

BATCH_SIZE = 128
lr = 0.05
# lr_table = {15 : 0.001, 30 : 0.0001}
lr_table = {32000 : 0.01, 48000 : 0.001}

data = load_cifar10_record(BATCH_SIZE)
n_training_samples = data[0].shape[0] if isinstance(data, np0.ndarray) else 50000 # TODO

optimizer_settings = {
  'initial_lr'   : lr,
# 'lr_scheduler' : AtEpochScheduler(lr, lr_table, n_training_samples, BATCH_SIZE), # TODO
  'lr_scheduler' : AtIterationScheduler(lr, lr_table),
  'optimizer'    : 'SGD',
  'args'         : {'momentum' : 0.9},
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 40,
  initializer = Xavier(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

identifier = 'triple-state-%s-residual-network' % MODE
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
