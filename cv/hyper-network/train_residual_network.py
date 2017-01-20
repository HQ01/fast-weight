import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

MODES = {'mode' : 'normal'}
# MODES = {'mode' : 'weight-sharing'}
# MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'batch_size' : BATCH_SIZE}
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

# TODO
if MODES['mode'] is 'normal':
  BATCH_SIZE = 128
  lr = 0.1
  lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}
  lr_scheduler = AtIterationScheduler(lr, lr_table)
  epochs = 150
elif MODES['mode'] is 'weight-sharing':
  BATCH_SIZE = 64
  lr = 0.01
  lr_table = {40 : 0.001, 50 : 0.0001}
  lr_scheduler = AtEpochScheduler(lr, lr_table, 50000, BATCH_SIZE)
  epochs = 60
elif MODES['mode'] is 'hyper': pass

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
  epochs = epochs,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

if MODES['mode'] is 'normal' or MODES['mode'] is 'normal' is 'weight-sharing':
  identifier = 'triple-state-%s-residual-network-%d' % (MODES['mode'], N)
elif MODES['mode'] is 'hyper':
  identifier = 'triple-state-%s-residual-network-%d-%s-embedding' % (MODES['mode'], N, MODES['embedding'])
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
