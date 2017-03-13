import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _attention_network(network):
  network = _normalized_convolution(X=network, n_filters=8, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=1)
  return network

def _softmax(scores):
  score_array = layers.concatenate(X=scores, n_inputs=len(scores), axis=1)
  probabilities = layers.softmax_activation(score_array)
  probability_list = layers.slice_channels(X=probabilities, n_outputs=len(scores))
  return probability_list
  
def _read(memory, settings):
  scores = tuple(_attention_network(slot) for slot in memory)
  if settings['probability'] is 'softmax': probabilities = _softmax(scores)
  attended_memory = 0
  for index, memory_slot in enumerate(memory):
    probability = layers.reshape(probabilities[index], (0, 1, 1, 1))
    attended_memory += layers.broadcast_multiply(probability, memory_slot)
  return attended_memory

_n_attended_memory_module = 0
def _attended_memory_module(network, settings):
  kwargs = {key : value for key, value in settings['convolution_settings'].items()}
  global _n_attended_memory_module
  prefix = 'attended_memory_module%d' % _n_attended_memory_module
  if settings['weight_sharing']:
    kwargs['weight'] = layers.variable('%s_weight' % prefix)
    kwargs['bias'] = layers.variable('%s_bias' % prefix)
  memory = [network]
  for index in range(settings['n_layers']):
    kwargs['X'] = _read(memory, settings)
    network = _normalized_convolution(**kwargs)
    memory.append(network)
  _n_attended_memory_module += 1
  return network

def attended_memory_network(settings):
  network = layers.variable('data')
  for module_settings in settings:
    if module_settings['operator'] is 'attended_memory_module':
      network = _attended_memory_module(network, module_settings['settings'])
    else:
      args = module_settings.get('args', tuple())
      kwargs = {key : value for key, value in module_settings.get('kwargs', {}).items()}
      if args: args = (network,) + args
      else: kwargs['X'] = network
      network = getattr(layers, module_settings['operator'])(*args, **kwargs)
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10)
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

if __name__ is '__main__':
  settings = (
    {
      'operator' : 'convolution',
      'kwargs' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
    },
    {
      'operator' : 'pooling',
      'kwargs' : {'mode' : 'maximum', 'kernel_shape' : (2, 2), 'stride' : (2, 2), 'pad' : (0, 0)},
    },
    {
      'operator' : 'attended_memory_module',
      'settings' : {
        'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
        'n_layers' : 3,
        'probability' : 'softmax',
        'weight_sharing' : True,
      },
    }
  )
  network = attended_memory_network(settings)
  print output_shape(network, data=(10000, 16, 32, 32))

  '''
  settings = {
    'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
    'n_layers' : 3,
    'probability' : 'softmax',
    'weight_sharing' : True,
  }
  network = layers.variable('data')
  network = _attended_memory_module(network, settings)
  print output_shape(network, data=(10000, 16, 32, 32))
  '''
  '''
  N = 16
  inputs = tuple(layers.variable('data%d' % index) for index in range(N))
  input_shapes = {'data%d' % index : (10, 1) for index in range(N)}
  outputs = _softmax(inputs)
  for index in range(N):
    print output_shape(outputs[index], **input_shapes)
  '''
