import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

network = triple_state_residual_network(1, mode='normal')

lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}

BATCH_SIZE = 128
data = load_cifar10_record(BATCH_SIZE)

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
  epochs = int(sys.argv[1]),
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

identifier = 'triple-state-transitory-residual-network'
pickle.dump(info, open('info/%s' % identifier, 'wb'))

parameters, states = solver.export_parameters()
parameters = {key : value for key, value in parameters.items() if 'transition' in key}
states = {key : value for key, value in states.items() if 'transition' in key}
pickle.dump((parameters, states), open('parameters/%s' % identifier, 'wb'))
