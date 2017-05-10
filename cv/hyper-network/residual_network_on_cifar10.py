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

def _module(network, n_filters, n_layers):
  for index in range(n_layers):
    identity = network
    network = _normalized_convolution(network, n_filters=n_filters)
    network = _normalized_convolution(network, n_filters=n_filters)
    network += identity

  return network

def _transit(network, n_filters):
  '''
  identity = \
      _convolution(X=network, n_filters=n_filters, kernel_shape=(1, 1), stride=(2, 2), pad=(0, 0))
  '''
  identity = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  identity = _convolution(X=identity, n_filters=n_filters, kernel_shape=(1, 1), pad=(0, 0))

  network = _normalized_convolution(network, n_filters=n_filters, stride=(2, 2))
  network = _normalized_convolution(network, n_filters=n_filters)

  return identity + network

def build_network(n_layers):
  network = layers.variable('data')
  network = _convolution(X=network, n_filters=16)

  for n_filters in (16, 32):
    network = _module(network, n_filters, n_layers)
    network = _transit(network, n_filters * 2)
  
  network = _module(network, 64, n_layers)
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)

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

  from mx_initializers import PReLUInitializer
  initializer = PReLUInitializer()

  from mx_solver import MXSolver
  solver = MXSolver(
    batch_size         = args.batch_size,
    devices            = (0, 1, 2, 3),
    epochs             = 150,
    initializer        = initializer,
    optimizer_settings = optimizer_settings,
    symbol             = network,
    verbose            = True,
  )

  from data_utilities import load_cifar10_record
  data = load_cifar10_record(args.batch_size)

  info = solver.train(data)

  postfix = '-' + args.postfix if args.postfix else ''
  identifier = 'residual-network-on-cifar-10-%d%s' % (args.n_layers, postfix)

  import cPickle as pickle
  pickle.dump(info, open('info/%s' % identifier, 'wb'))
  parameters = solver.export_parameters()
  pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
