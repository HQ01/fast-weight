import mx_layers as layers

def _convolution(**kwargs):
  defaults = {'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
  defaults.update(kwargs)
  return layers.convolution(**defaults)

_n_rnn_linearities = 0
def _rnn_convolution(X, n_filters, weight):
  global _n_rnn_linearities
  id = 'rnn_linearity%d' % _n_rnn_linearities
  _n_rnn_linearities += 1
  return \
    _convolution(X=X, n_filters=n_filters, kernel_shape=(1, 1), pad=(0, 0), weight=weight, id=id)

def elman(X, n_filters, cache):
  time = cache.setdefault('time', -1)
  cache['time'] += 1

  WX = cache.setdefault('WX', layers.variable('X_weight'))
  WH = cache.setdefault('WH', layers.variable('H_weight'))
  bias = cache.setdefault('bias', layers.variable('elman_bias', shape=(1, n_filters, 1, 1)))

  network = _rnn_convolution(X, n_filters, WX) + \
    (_rnn_convolution(cache['h'], n_filters, WH) if 'h' in cache else 0)

  network = layers.broadcast_plus(network, bias)
# network = layers.batch_normalization(network, fix_gamma=False, id='ElmanBN%d' % time)

  cache['h'] = layers.tanh(network)

  return cache

def lstm(X, n_filters, cache):
  WX = cache.setdefault('WX', layers.variable('X_weight'))
  WH = cache.setdefault('WH', layers.variable('H_weight'))
  bias = cache.setdefault('bias', layers.variable('lstm_bias', shape=(1, n_filters * 4, 1, 1)))
  
  network = _rnn_convolution(X, n_filters * 4, WH) + \
    (_rnn_convolution(cache['h'], n_filters * 4, WH) if 'h' in cache else 0)
  network = layers.broadcast_plus(network, bias)

  group = layers.slice(X=network, axis=1, n_outputs=4)
  i = layers.sigmoid(group[0])
  f = layers.sigmoid(group[1])
  o = layers.sigmoid(group[2])
  g = layers.tanh(group[3])

  cache['c'] = f * cache.get('c', 0) + i * g
  cache['h'] = o * layers.tanh(cache['c'])
  
  return cache

def _normalized_convolution(network, **kwargs):
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)
  network = _convolution(X=network, **kwargs)
  return network

def _traced_module(network, rnn, n_filters, n_layers):
  rnn_cache = {}

  for index in range(n_layers):
    identity = network

    residual = _normalized_convolution(network, n_filters=n_filters)
    residual = _normalized_convolution(residual, n_filters=n_filters)
    trace = layers.terminate_gradient(residual)
    rnn_cache = globals()[rnn](trace, n_filters, rnn_cache)

    network = identity + residual

  return network, rnn_cache

def _module(network, n_filters, n_layers):
  for index in range(n_layers):
    identity = network
    residual = _normalized_convolution(network, n_filters=n_filters)
    residual = _normalized_convolution(residual, n_filters=n_filters)
    network = identity + residual

  return network

def _transit(network, n_filters):
  identity = \
    _convolution(X=network, n_filters=n_filters, kernel_shape=(1, 1), stride=(2, 2), pad=(0, 0))

  network = _normalized_convolution(network, n_filters=n_filters, stride=(2, 2))
  network = _normalized_convolution(network, n_filters=n_filters)

  return identity + network

def build_network(args):
  network = layers.variable('data')
  network = _convolution(X=network, n_filters=16)

  for n_filters in (16, 32):
    network = _module(network, n_filters, args.n_layers)
    network = _transit(network, n_filters * 2)
  
# network = _module(network, 64, args.n_layers)
  _, rnn_cache = _traced_module(network, args.rnn, 64, args.n_layers)

# network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.batch_normalization(rnn_cache['h'], fix_gamma=False, id='BN')
  network = layers.ReLU(network)

  network = layers.pooling(X=rnn_cache['h'], mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10, id='linear')
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

  return network

import mxnet as mx
class RNNInitializer(object):
  def __call__(self, id, array):
    if id == 'H_weight':
      if args.rnn == 'elman':
        array[:] = mx.nd.zeros(array.shape)
        for i in range(array.shape[0]):
          array[i][i][:] = 1
      elif args.rnn == 'lstm':
        mx.initializer.Xavier()(id, array)

    elif 'weight' in id: mx.initializer.Xavier()(id, array)
    elif 'bias' in id: array[:] = 0

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--initial_lr', type=float, default=0.1)
  parser.add_argument('--network_path', type=str, default='')
  parser.add_argument('--n_layers', type=int, required=True)
  parser.add_argument('--postfix', type=str, default='')
  parser.add_argument('--rnn', type=str, required=True)
  args = parser.parse_args()

  network = build_network(args)

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

  import cPickle as pickle
  path = args.network_path if args.network_path else \
    'parameters/residual-network-on-cifar-10-%d' % args.n_layers
  parameters, states = pickle.load(open(path, 'rb'))
  parameters.update(states)

  from mx_initializers import HybridInitializer
  initializer = HybridInitializer(parameters, RNNInitializer())

  from mx_solver import MXSolver
  solver = MXSolver(
    batch_size         = args.batch_size,
    devices            = (0, 1, 2, 3),
    epochs             = 1,
    initializer        = initializer,
    optimizer_settings = optimizer_settings,
    symbol             = network,
    verbose            = True,
  )

  from data_utilities import load_cifar10_record
  data = load_cifar10_record(args.batch_size)

  info = solver.train(data)

  postfix = '-' + args.postfix if args.postfix else ''
  identifier = '%s-rnn-on-residual-network-on-cifar-10-%d%s' % (args.rnn, args.n_layers, postfix)
  print identifier

  print 'from'
  pickle.dump(info, open('info/%s' % identifier, 'wb'))
  print 'to'
  parameters = solver.export_parameters()
  pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
