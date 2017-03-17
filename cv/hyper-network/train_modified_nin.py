import cPickle as pickle

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from modified_nin import nin

settings = {}
settings['transition_mode'] = 'convolution + pooling'
# settings['transition_mode'] = 'pooling + pooling'
# settings['transition_mode'] = 'stochastic_pooling'
network = nin(settings)

BATCH_SIZE = 128

lr = 0.05
lr_table = {24000 : lr * 0.1, 48000 : lr * 0.01}

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

identifier = 'modified-nin-%s' % (settings['transition_mode'])
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
