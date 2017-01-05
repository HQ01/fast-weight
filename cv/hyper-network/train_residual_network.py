import cPickle as pickle

from lr_scheduler import FactorScheduler
from facility import load_cifar10
from mxnet.initializer import Xavier
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

data = load_cifar10(center=True, rescale=True, reshape=True)

N = 3
MODE = 'normal'
network = triple_state_residual_network(N, MODE)

BATCH_SIZE = 64
lr = 0.01

optimizer_settings = {
  'initial_lr' : lr,
  'optimizer'  : 'Adam',
  'lr_scheduler' : FactorScheduler(lr, 0.99, data[0].shape[0] // BATCH_SIZE), # TODO
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 50,
  initializer = Xavier(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = False,
)

info = solver.train(data)

identifier = 'triple-state-%s-residual-network' % MODE
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
