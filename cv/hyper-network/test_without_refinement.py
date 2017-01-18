import cPickle as pickle
import sys

N = int(sys.argv[1])
refined_parameters, refined_states = pickle.load(open('parameters/triple-state-refined-residual-network-%d' % N, 'rb'))
transitory_parameters = {key : value for key, value in refined_parameters.items() if 'transition' in key}
transitory_states = {key : value for key, value in refined_states.items() if 'transition' in key}

network = triple_state_residual_network(1, mode='normal')
