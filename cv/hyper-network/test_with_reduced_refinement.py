import cPickle as pickle
import sys
import mxnet as mx
from data_utilities import load_cifar10_record
from mx_utilities import feed_forward, to_mx_arrays
from residual_network import triple_state_residual_network

N = 20
i = int(sys.argv[1])

def map_layers(parameters):
  p = {}
  for key, value in parameters.items():
    if 'batchnorm' in key:
      start, end = len('batchnorm'), key.index('_')
      index = int(key[start : end])
      if index > 1:
        index -= 2
        transition, refinement = (index + 1) // ((N - 1) * 2), (index + 1) % ((N - 1) * 2)
        if refinement < i:
          to_index = transition * 2 * i + refinement + 2
          to_key = key.replace(str(index + 2), str(to_index))
          p[to_key] = value
      else: p[key] = value
    else: p[key] = value
  return p

refined_parameters, refined_states = pickle.load(open('parameters/triple-state-refined-residual-network-%d' % N, 'rb'))
for key in refined_parameters.keys():
  if 'batchnorm' in key: print key
print '#' * 30
for key in map_layers(refined_parameters).keys():
  if 'batchnorm' in key: print key

refined_parameters = to_mx_arrays(refined_parameters)
refined_states = to_mx_arrays(refined_states)

network = triple_state_residual_network(i, mode='weight-sharing')

'''
model = feed_forward(
  symbol           = network,
  parameters       = refined_parameters,
  auxiliary_states = refined_states,
  extra_parameters = True,
  context          = mx.gpu(0),
)
BATCH_SIZE = 1000
_, _, data = load_cifar10_record(BATCH_SIZE)
accuracy = model.score(data)
'''
