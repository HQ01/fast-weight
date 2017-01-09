# TODO untrainable transition weights
import cPickle as pickle
import numpy as np0
import sys

from lr_scheduler import AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_initializer import HybridInitializer, PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

BATCH_SIZE = 128
MODES = {'mode' : 'weight-sharing'}
# MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'batch_size' : BATCH_SIZE}
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

BATCH_SIZE = 128
data = load_cifar10_record(BATCH_SIZE)

initializer = HybridInitializer(
  pickle.load(open('parameters/triple-state-transitory-residual-network', 'rb')),
  PReLUInitializer()
)
'''
initializer = PReLUInitializer() 
'''

lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}

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
  epochs = 150,
  initializer = initializer,
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

identifier = 'triple-state-refined-residual-network'
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
