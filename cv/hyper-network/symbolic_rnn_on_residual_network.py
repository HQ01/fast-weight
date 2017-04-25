import mx_layers as layers
import numpy as np

def _convolution(**kwargs):
  defaults = {'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1), 'no_bias' : True}
  defaults.update(kwargs)
  return layers.convolution(**defaults)

_n_rnn_linearities = 0
def _rnn_linearity(X, D, weight):
  global _n_rnn_linearities
  id = 'rnn_linearity%d' % _n_rnn_linearities
  _n_rnn_linearities += 1

  return layers.fully_connected(X=X, n_hidden_units=D, weight=weight, no_bias=True, id=id)

def elman(X, D, cache):
  time = cache.setdefault('time', -1)
  cache['time'] += 1

  WX = cache.setdefault('WX', layers.variable('X_weight'))
  WH = cache.setdefault('WH', layers.variable('H_weight'))
  bias = cache.setdefault('bias', layers.variable('elman_bias', shape=(1, D)))

  network = _rnn_linearity(X, D, WX) + (_rnn_linearity(cache['h'], D, WH) if 'h' in cache else 0)
  network = layers.broadcast_plus(network, bias)
  cache['h'] = layers.tanh(network)

  return cache

def lstm(X, D, cache):
  time = cache.setdefault('time', -1)
  cache['time'] += 1

  WX = cache.setdefault('WX', layers.variable('X_weight'))
  WH = cache.setdefault('WH', layers.variable('H_weight'))
  bias = cache.setdefault('bias', layers.variable('lstm_bias', shape=(1, D * 4)))

  network = _rnn_linearity(X, D * 4, WX) + (_rnn_linearity(cache['h'], D * 4, WH) if 'h' in cache else 0)
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

def _traced_module(network, n_filters, n_layers):
  group = []

  for index in range(n_layers):
    identity = network

    residual = _normalized_convolution(network, n_filters=n_filters)
    residual = _normalized_convolution(residual, n_filters=n_filters)

    trace = layers.terminate_gradient(residual)
    trace = layers.ReLU(trace)
    trace = layers.flatten(trace)
    group.append(trace)

    network = identity + residual

  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)

  network = layers.pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = layers.flatten(network)
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.fully_connected(X=network, n_hidden_units=10)
  network = layers.terminate_gradient(network)
  group.append(network)

  return layers.group(group)

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

def build_resnet(args):
  network = layers.variable('data')
  network = _convolution(X=network, n_filters=16)

  for n_filters in (16, 32):
    network = _module(network, n_filters, args.n_layers)
    network = _transit(network, n_filters * 2)
  
  return _traced_module(network, 64, args.n_layers)

def build_rnn(args):
  rnn_cache = {}

  for i in range(args.n_layers):
    X = layers.variable('data%d' % i)
    rnn_cache = globals()[args.rnn](X, args.n_hidden_units, rnn_cache)

  network = layers.fully_connected(X=rnn_cache['h'], n_hidden_units=10, id='linear')
  loss = layers.linear_regression_loss(network, id='criterion')
# network = layers.softmax_loss(prediction=network, normalization='batch', id='criterion')

  return network, loss

def cross_entropy_loss(predictions, labels, n_classes=10):
  predictions = mx.nd.softmax(predictions, axis=1).asnumpy()
  labels = mx.nd.one_hot(labels, n_classes).asnumpy()
  loss = - np.sum(np.log(predictions) * labels) / len(labels)
  return loss

def classification_accuracy(predictions, labels):
  predictions, labels = predictions.asnumpy(), labels.asnumpy()
  differences = np.argmax(predictions, axis=1) - labels
  return 1 - np.count_nonzero(differences) / float(len(labels))

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--n_epochs', type=int, default=32)
  parser.add_argument('--gpu_index', type=int, default=0)
  parser.add_argument('--initial_lr', type=float, default=0.1)
  parser.add_argument('--n_hidden_units', type=int, required=True)
  parser.add_argument('--n_layers', type=int, required=True)
  parser.add_argument('--network_path', type=str, default='')
  parser.add_argument('--path', type=str, default='')
  parser.add_argument('--postfix', type=str, default='')
  parser.add_argument('--rnn', type=str, required=True)
  args = parser.parse_args()

  import mxnet as mx
  context = mx.gpu(args.gpu_index)

  # resnet
  import cPickle as pickle
  path = args.path if args.path else \
    'parameters/residual-network-on-cifar-10-%d' % args.n_layers
  resnet_args, resnet_states = pickle.load(open(path, 'rb'))

  to_array = lambda array : mx.nd.array(array, context)
  resnet_args = dict(zip(resnet_args, map(to_array, resnet_args.values())))
  resnet_states = dict(zip(resnet_states, map(to_array, resnet_states.values())))

  resnet_args['data'] = mx.nd.zeros((args.batch_size, 3, 32, 32), context)

  resnet = build_resnet(args)

  resnet_executor = resnet.bind(context, resnet_args, aux_states=resnet_states)

  # rnn
  rnn, rnn_loss = build_rnn(args)
  rnn_args = rnn.list_arguments()
  _, data_shapes, _ = resnet.infer_shape(data=(args.batch_size, 3, 32, 32))
  logit_shape = data_shapes[-1]
  data_shapes = {'data%d' % i : shape for i, shape in enumerate(data_shapes[:-1])}
  rnn_arg_shapes, _, _ = rnn.infer_shape(**data_shapes)

  rnn_arg_dict = {}

  from mxnet.initializer import Orthogonal, Xavier
  orthogonal = Orthogonal()
  xavier = Xavier()
  for arg, shape in zip(rnn_args, rnn_arg_shapes):
    rnn_arg_dict[arg] = mx.nd.zeros(shape, context)

    if args.rnn == 'elman' and 'H_weight' in arg:
      orthogonal(arg, rnn_arg_dict[arg])
    elif 'weight' in arg: xavier(arg, rnn_arg_dict[arg])

  rnn_executor = rnn.bind(context, rnn_arg_dict)

  rnn_args_grad = {
    arg : mx.nd.zeros(shape, context) \
      for arg, shape in zip(rnn_args, rnn_arg_shapes) if 'data' not in arg
  }

  rnn_loss_arg_dict = rnn_arg_dict.copy()
  rnn_loss_arg_dict['criterion_label'] = mx.nd.zeros(logit_shape, context)
  rnn_loss_executor = rnn_loss.bind(context, rnn_loss_arg_dict, rnn_args_grad)

  mean_dict = {key : mx.nd.zeros(value.shape, context) for key, value in rnn_args_grad.items()}
  variance_dict = {key : mx.nd.ones(value.shape, context) for key, value in rnn_args_grad.items()}

  from data_utilities import load_cifar10_record
  training_data, _, _ = load_cifar10_record(args.batch_size)

  for epoch in range(args.n_epochs):
    training_data.reset()
    for iteration, batch in enumerate(training_data):
      resnet_executor.arg_dict['data'][:] = batch.data[0]
      outputs = resnet_executor.forward()

      for index, output in enumerate(outputs[:-1]):
        rnn_loss_executor.arg_dict['data%d' % index][:] = output
      rnn_loss_executor.arg_dict['criterion_label'][:] = outputs[-1]
      outputs = rnn_loss_executor.forward(is_train=True)
      rnn_loss_executor.backward()

      if (iteration + 1) % 100 == 0:
        outputs = rnn_executor.forward()
        accuracy = classification_accuracy(outputs[0], batch.label[0])
        loss = cross_entropy_loss(outputs[0], batch.label[0])
        print 'iteration %d accuracy %f loss %f' % (iteration + 1, accuracy, loss)

      from mxnet.ndarray import adam_update as update
      for key, value in rnn_args_grad.items():
        if value is None: continue

        array = rnn_arg_dict[key]
        mean = mean_dict[key]
        variance = variance_dict[key]
        update(array, value, mean, variance, 1e-5, out=array)

  postfix = '-' + args.postfix if args.postfix else ''
  identifier = '%s-rnn-on-residual-network-on-cifar-10-%d%s' % (args.rnn, args.n_layers, postfix)

  '''
  pickle.dump(info, open('info/%s' % identifier, 'wb'))
  parameters = solver.export_parameters()
  pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
  '''
