import cPickle as pickle

from data_utilities import load_cifar10
from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from GPU_utility import GPU_availability

import mx_layers as layers

N_HIDDEN_UNITS = 1536
N_LAYERS = 3
network = layers.variable('data')
for index in range(N_LAYERS):
  network = layers.fully_connected(X=network, n_hidden_units=N_HIDDEN_UNITS)
  network = layers.ReLU(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

BATCH_SIZE = 64

lr = 0.1
lr_table = {}

lr_scheduler = AtIterationScheduler(lr, lr_table)

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = GPU_availability()[:1],
  epochs = 150,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_cifar10(center=True, rescale=True)
info = solver.train(data)

identifier = 'cifar-baseline-mlp-%d-%d' % (N_LAYERS, N_HIDDEN_UNITS)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
