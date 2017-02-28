import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mxnet.initializer import Xavier, MSRAPrelu
from mxnet.visualization import print_summary
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from generalized_dense_network import dense_network

def identity(cache):
  return cache
def previous_n_activations(n_activations):
  def connect(cache):
    memory = []
    for index, activation in enumerate(reversed(cache)):
      if index < n_activations: memory.append(activation)
      else: memory.append(0 * activation)
    return memory
  return connect
  
N_LAYERS = int(sys.argv[1])
CONNECT = sys.argv[2]
CONNECT = identity
# CONNECT = previous_n_activations(1)
SETTINGS = ({'N_LAYERS' : N_LAYERS, 'CONNECT' : CONNECT},) * 3
network = dense_network(SETTINGS)

BATCH_SIZE = 128
EPOCHS = 150
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
  batch_size         = BATCH_SIZE,
  devices            = (0, 1, 2, 3),
  epochs             = EPOCHS,
  initializer        = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol             = network,
  verbose            = True,
)

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'generalized-dense-network-%d-layers' % (N_LAYERS)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
