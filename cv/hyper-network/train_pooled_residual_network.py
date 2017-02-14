import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

refining_times = int(sys.argv[2])
pooling_times = int(sys.argv[3])
settings = {'mode' : sys.argv[1], 'times' : refining_times, 'pooling_times' : pooling_times}
network = triple_state_residual_network(settings)
# print_summary(network)

BATCH_SIZE = 128
lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}
lr_scheduler = AtIterationScheduler(lr, lr_table)
epochs = 1

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

identifier = 'triple-state-%s-residual-network-refining-%d-pooling-%d' % (settings['mode'], refining_times, pooling_times)
if settings['mode'] is 'hyper':
  identifier += '-embedding-%s' % settings['embedding']
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
