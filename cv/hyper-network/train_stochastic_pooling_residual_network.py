import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from stochastic_pooling_residual_network import stochastic_pooling_residual_network

settings = {
  'n_layers'     : int(sys.argv[1]),
  'pooling_mode' : sys.argv[2],
  'p'            : float(sys.argv[3]),
}
network = stochastic_pooling_residual_network(settings)

BATCH_SIZE = 128

lr = 0.1
lr_table = {27000 : lr * 0.1, 48000 : lr * 0.01}

'''
lr = 0.003
lr_table = {10000 : 0.001, 20000 : 0.0005}
'''

lr_scheduler = AtIterationScheduler(lr, lr_table)

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

'''
optimizer_settings = {
  'args'         : {},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'Adam',
}
'''

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

identifier = 'stochastic-%s-pooling-residual-network-%d-p-%f' % (settings['pooling_mode'], settings['n_layers'], settings['p'])
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
