import cPickle as pickle
import numpy as np
import sys

def compare(T, R):
  for key, value in T.items():
    difference = np.max(np.abs(value - R[key]))
    print key, difference

N = int(sys.argv[1])
transitory_parameters, transitory_states = pickle.load(open('parameters/triple-state-transitory-residual-network', 'rb'))
refined_parameters, refined_states = \
  pickle.load(open('parameters/triple-state-refined-residual-network-%d-constant-transition' % N, 'rb'))

compare(transitory_parameters, refined_parameters)
compare(transitory_states, refined_states)
