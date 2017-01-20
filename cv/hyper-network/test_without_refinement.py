import cPickle as pickle
import sys
import mxnet as mx
from data_utilities import load_cifar10_record
from mx_utilities import feed_forward, to_mx_arrays
from residual_network import triple_state_residual_network

accuracy_table = []
for N in range(21):
  refined_parameters, refined_states = pickle.load(open('parameters/triple-state-refined-residual-network-%d' % N, 'rb'))
  transitory_parameters = to_mx_arrays({key : value for key, value in refined_parameters.items() if 'transition' in key})
  transitory_states = to_mx_arrays({key : value for key, value in refined_states.items() if 'transition' in key})

  network = triple_state_residual_network(1, mode='weight-sharing')
  model = feed_forward(
    symbol           = network,
    parameters       = transitory_parameters,
    auxiliary_states = transitory_states,
    context          = mx.gpu(0),
  )
  BATCH_SIZE = 1000
  _, _, data = load_cifar10_record(BATCH_SIZE)
  accuracy = model.score(data)
  accuracy_table.append(accuracy)
pickle.dump(accuracy_table, open('test_accuracy_without_refinement', 'wb'))
