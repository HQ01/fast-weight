# using a plain network

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--n_plain_layers', type=int, default=3)
# parser.add_argument('--n_plain_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, default='')
configs = parser.parse_args()

import mx_layers as layers

_convolution = lambda X : layers.convolution(X=X, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
network = layers.variable('data')
for index in range(3):
  network = _convolution(network)
  network = layers.ReLU(network)
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

shared_weight = layers.variable('shared_weight')
shared_gamma = layers.variable('shared_gamma')
shared_beta = layers.variable('shared_beta')

_convolution = lambda X : layers.convolution(
  X=X, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1), weight=shared_weight, no_bias=True
)
for index in range(configs.n_plain_layers):
  network = layers.batch_normalization(network, beta=shared_beta, gamma=shared_gamma, fix_gamma=False)
  network = layers.ReLU(network)
  network = _convolution(network)

network = layers.pooling(X=network, mode='average', kernel_shape=(7, 7), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

import mxnet as mx
context = mx.gpu(configs.gpu_index)
arg_shapes, _, aux_state_shapes = network.infer_shape(data=(configs.batch_size, 1, 56, 56))
args = network.list_arguments()
args = {arg : mx.nd.zeros(shape, context) for arg, shape in zip(args, arg_shapes)}
args_grad = {arg : mx.nd.zeros(shape, context) for arg, shape in zip(args, arg_shapes)}
aux_states = network.list_auxiliary_states()
aux_states = {aux_state : mx.nd.zeros(shape, context) for aux_state, shape in zip(aux_states, aux_state_shapes)}

from mx_initializers import PReLUInitializer
initializer = PReLUInitializer()
for key, value in args.items():
  if key != 'data' and key != 'softmax_label': initializer(key, value)
for key, value in aux_states.items(): initializer(key, value)

executor = network.bind(context, args, args_grad, aux_states=aux_states)

_convolution = lambda X : layers.convolution(X=X, n_filters=16, kernel_shape=(9, 9), stride=(1, 1), pad=(4, 4))
network = layers.variable('data')
for index in range(3):
  network = _convolution(network)
  network = layers.ReLU(network)
  # TODO impact of pooling
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

shared_weight = layers.variable('shared_weight')
shared_gamma = layers.variable('shared_gamma')
shared_beta = layers.variable('shared_beta')

_convolution = lambda X : layers.convolution(
  X=X, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2), weight=shared_weight, no_bias=True
)
for index in range(configs.n_plain_layers):
  network = layers.batch_normalization(network, beta=shared_beta, gamma=shared_gamma, fix_gamma=False)
  network = layers.ReLU(network)
  network = _convolution(network)

network = layers.pooling(X=network, mode='average', kernel_shape=(7, 7), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

args = {key : mx.nd.zeros_like(value) for key, value in args.items()}
aux_states = {key : mx.nd.zeros_like(value) for key, value in aux_states.items()}
inference_executor = network.bind(context, args, aux_states=aux_states)

from data_utilities import load_mnist
original = load_mnist(path='stretched_canvas_mnist', scale=1, shape=(1, 56, 56))
stretched = load_mnist(path='stretched_mnist', scale=1, shape=(1, 56, 56))

from mxnet.io import NDArrayIter
training_data = NDArrayIter(original[0], original[1], configs.batch_size, True)
validation_data = NDArrayIter(stretched[2], stretched[3], configs.batch_size, False, 'discard')
test_data = NDArrayIter(stretched[4], stretched[5], configs.batch_size, False, 'discard')

unpack = lambda batch : (batch.data[0], batch.label[0])
for epoch in range(configs.n_epochs):
  for batch in training_data:
    data, labels = unpack(batch)
    print 'unpacked'
    executor.arg_dict['data'][:] = data
    executor.arg_dict['softmax_label'][:] = labels
    executor.forward(is_train=True)
    executor.backward()

    for key, value in args_grad.items():
      args[key] -= configs.lr * value

  for batch in validation_data:
    data, labels = unpack(batch)
    arg_dict = inference_executor.arg_dict
    arg_dict['data'][:] = data
    arg_dict['softmax_label'][:] = labels
    for key, value in executor.items():
      if 'convolution' in key and 'weight' in key:
        weight = value.asnumpy()
        rescaled = np.zeros(arg_dict[key].shape)
        IN, OUT, L, L = weight.shape
        for i in range(IN):
          for j in range(OUT):
            rescaled[i][j] = imresize(weight[i][j], (L * 2 - 1, L * 2 - 1))
        arg_dict[key][:] = rescaled
      else:
        inference_executor.arg_dict[key][:] = value
    inference_executor.forward(is_train=True) # using batch statistics
