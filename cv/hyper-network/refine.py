# TODO untrainable transition weights
import cPickle as pickle
import numpy as np0
import sys
import mxnet as mx

from lr_scheduler import AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_initializer import HybridInitializer, PReLUInitializer
from mx_solver import MXSolver

from residual_network import triple_state_residual_network

BATCH_SIZE = 128
MODES = {'mode' : 'weight-sharing'}
# MODES = {'mode' : 'hyper', 'embedding' : 'parameter'}
# MODES = {'mode' : 'hyper', 'embedding' : 'feature_map', 'batch_size' : BATCH_SIZE}
constant_transition = True
MODES['constant_transition'] = constant_transition
N = int(sys.argv[1])
network = triple_state_residual_network(N, **MODES)

BATCH_SIZE = 128
data = load_cifar10_record(BATCH_SIZE)

transitory_parameters, transitory_states = \
  pickle.load(open('parameters/triple-state-transitory-residual-network', 'rb'))

initializer = HybridInitializer(
  transitory_parameters,
  PReLUInitializer()
)

lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : AtIterationScheduler(lr, lr_table),
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

constant_parameters = transitory_parameters if constant_transition else None

solver = MXSolver(
  auxiliary_states    = transitory_states,
  batch_size          = BATCH_SIZE,
  constant_parameters = constant_parameters,
  devices             = (0, 1, 2, 3),
  epochs              = 150,
  initializer         = initializer,
  optimizer_settings  = optimizer_settings,
  symbol              = network,
  verbose             = True,
)

info = solver.train(data)

'''
identifier = 'triple-state-refined-residual-network-%d' % N
if constant_transition: identifier += '-constant-transition'
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
'''
