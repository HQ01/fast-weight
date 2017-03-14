import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10, load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from GPU_utility import GPU_availability

from dropping_out_mlp import dropping_out_mlp

N_HIDDEN_UNITS = 3072
N_LAYERS = 3
settings = {
  'layer_settings' : (
    {'n_hidden_units' : N_HIDDEN_UNITS, 'p' : 0.5},
  ) * N_LAYERS,
  'n_classes' : 10,
}
network = dropping_out_mlp(settings)

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

# data = load_cifar10(center=True, rescale=True)
data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'dropping-out-mlp-%d-%d' % (N_LAYERS, N_HIDDEN_UNITS)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
