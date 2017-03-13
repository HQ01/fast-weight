import cPickle as pickle
import gzip
import sys

from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from data_utilities import load_mnist

from rescaled_mnist_baseline_network import naive_network
N = int(sys.argv[1])
weight_sharing = (sys.argv[2] is 'weight-sharing')
network = naive_network(N, weight_sharing)

BATCH_SIZE = 128
lr = 0.1
lr_table = {10000 : 0.01}
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
  epochs = 50,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_mnist(path='rescaled_mnist', shape=(1, 42, 42))
info = solver.train(data)

identifier = 'rescaled-mnist-baseline-network-%d-%s' % (N, sys.argv[2])
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
