# TODO check network architecture
import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

REFINING_TIMES = int(sys.argv[1])
# INTERVALS = (1, 2, 4, 8) # TODO INTERVALS being an argument
INTERVALS = map(int, sys.argv[2:])
settings = {
  'coefficients' : (1.0 / len(INTERVALS),) * len(INTERVALS),
  'intervals'    : INTERVALS,
  'mode'         : 'hybrid',
  'times'        : REFINING_TIMES,
}
network = triple_state_residual_network(settings)

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

POSTFIX = 'round1'
identifier = 'triple-state-%s-residual-network-refining-%d-interval-%s-%s' % \
  (settings['mode'], REFINING_TIMES, '-'.join(map(str, INTERVALS)), POSTFIX)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
