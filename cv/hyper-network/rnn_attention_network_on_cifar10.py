import mx_layers as layers

def _convolution(**kwargs):
  defaults = {'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
  defaults.update(kwargs)
  return layers.convolution(**defaults)

def _normalized_convolution(network, **kwargs):
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)
  network = _convolution(X=network, **kwargs)
  return network

def _rnn_convolution(X, n_filters, weight):
  return _convolution(X=X, n_filters=n_filters, kernel_shape=(1, 1), pad=(0, 0), weight=weight)

def _rnn(X, n_filters, parameters, memory):
  X_weight, h_weight, bias = parameters
  previous_h, previous_c = memory

  if previous_h is 0: array = _rnn_convolution(X, n_filters, X_weight)
  else: array = \
      _rnn_convolution(X, n_filters, X_weight) + _rnn_convolution(previous_h, n_filters, h_weight)
  array = layers.broadcast_plus(array, bias)

  group = layers.slice(X=array, axis=1, n_outputs=4)
  i = layers.sigmoid(group[0])
  f = layers.sigmoid(group[1])
  o = layers.sigmoid(group[2])
  g = layers.tanh(group[3])

  next_c = f * previous_c + i * g
  next_h = o * layers.tanh(next_c)
  memory = next_h, next_c

  return memory

def _read(n_filters, memory):
  h, c = memory
  return _convolution(X=h, n_filters=n_filters)

def _write(X, n_filters, parameters, memory):
  return _rnn(X, n_filters, parameters, memory)

n_modules = 0
def _rnn_attention_module(network, settings):
  global n_modules
  prefix = 'rnn_attention_module%d' % n_modules
  n_modules += 1

  n_filters = settings['convolution_settings']['n_filters']
  X_weight = layers.variable('%s_X_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  h_weight = layers.variable('%s_h_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  rnn_bias = layers.variable('%s_rnn_bias' % prefix, shape=(1, 4 * n_filters, 1, 1))
  rnn_parameters = (X_weight, h_weight, rnn_bias)
  memory = 0, 0

  kwargs = {key : value for key, value in settings['convolution_settings'].items()}

  if settings['weight_sharing']:
    # TODO
    kwargs['weight'] = layers.variable('%s_weight' % prefix)
    shared_gamma = layers.variable('shared_gamma')
    shared_beta = layers.variable('shared_beta')

  for index in range(settings['n_layers']):
    from_identity = network

    memory = _write(network, n_filters * 4, rnn_parameters, memory)
    from_rnn = _read(n_filters, memory)

    network = _normalized_convolution(network, **kwargs)
    network = _normalized_convolution(network, **kwargs)

    network += from_identity + from_rnn
#   network += from_rnn

  return network

def _transit(network, n_filters):
  identity = \
    _convolution(X=network, n_filters=n_filters, kernel_shape=(1, 1), stride=(2, 2), pad=(0, 0))

  network = _normalized_convolution(network, n_filters=n_filters, stride=(2, 2))
  network = _normalized_convolution(network, n_filters=n_filters)

  return identity + network

def build_network(n_layers):
  network = layers.variable('data')
  network = _convolution(X=network, n_filters=16)

  convolution_settings = {'n_filters' : None}
  settings = {'convolution_settings' : convolution_settings, 'n_layers' : args.n_layers, 'weight_sharing' : False}

  for n_filters in (16, 32):
    convolution_settings['n_filters'] = n_filters
    network = _rnn_attention_module(network, settings)
    network = _transit(network, n_filters * 2)

  convolution_settings['n_filters'] = 64
  network = _rnn_attention_module(network, settings)

  network = layers.pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = layers.flatten(network)
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.fully_connected(X=network, n_hidden_units=10)
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

  return network

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--initial_lr', type=float, default=0.1)
  parser.add_argument('--n_layers', type=int, required=True)
  parser.add_argument('--postfix', type=str, default='')
  parser.add_argument('--sharing', type=bool, default=False)
  args = parser.parse_args()

  network = build_network(n_layers=args.n_layers)

  from lr_scheduler import AtIterationScheduler
  lr_table = {32000 : args.initial_lr * 0.1, 48000 : args.initial_lr * 0.01}
  lr_scheduler = AtIterationScheduler(args.initial_lr, lr_table)

  optimizer_settings = {
    'args'         : {'momentum' : 0.9},
    'initial_lr'   : args.initial_lr,
    'lr_scheduler' : lr_scheduler,
    'optimizer'    : 'SGD',
    'weight_decay' : 0.0001,
  }

  from mx_solver import MXSolver
  from mx_initializer import PReLUInitializer
  solver = MXSolver(
    batch_size         = args.batch_size,
    devices            = (0, 1, 2, 3),
    epochs             = 150,
    initializer        = PReLUInitializer(),
    optimizer_settings = optimizer_settings,
    symbol             = network,
    verbose            = True,
  )

  from data_utilities import load_cifar10_record
  data = load_cifar10_record(args.batch_size)

  info = solver.train(data)

  postfix = '-' + args.postfix if args.postfix else ''
  identifier = 'rnn-attention-network-on-cifar-10-%d%s' % (args.n_layers, postfix)

  import cPickle as pickle
  pickle.dump(info, open('info/%s' % identifier, 'wb'))
  parameters = solver.export_parameters()
  pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
