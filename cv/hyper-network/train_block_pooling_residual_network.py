import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from block_pooling_residual_network import pooled_residual_network

N = 3
refining_times = map(int, sys.argv[1 : 1 + N])
pooling_times = map(int, sys.argv[1 + N : 1 + 2 * N])
settings = {'refining_times' : refining_times, 'pooling_times' : pooling_times}
network = pooled_residual_network(settings)

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

identifier = 'block-pooling-residual-network-refinement-%s-pooling-%s' % \
  ('-'.join(map(str, refining_times)), '-'.join(map(str, pooling_times)))
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
