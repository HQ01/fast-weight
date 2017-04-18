import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n_layers', type=int, required=True)
args = parser.parse_args()

from mx_layers import *

_convolution = lambda X : \
  convolution(X=X, n_filters=8, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True)

network = variable('data')
network = _convolution(network)
network = pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

classifier_weight = variable('classifier_weight')
classifier_bias = variable('classifier_bias')

_classify = lambda X : \
  fully_connected(X=network, n_hidden_units=10, weight=classifier_weight, bias=classifier_bias)

label = variable('labels')
symbols = []

for i in range(args.n_layers):
  network = batch_normalization(network)
  network = ReLU(network)
  network = _convolution(network)
  pooled = pooling(X=network, mode='average', kernel_shape=(28, 28), stride=(1, 1), pad=(0, 0))
  scores = _classify(pooled)
  loss = softmax_loss(prediction=scores, label=label, normalization='batch')
  symbols.append(loss)

loss_group = group(symbols)

arg_shapes, _, state_shapes = loss_group.infer_shape(data=(args.batch_size, 1, 56, 56))

import mxnet as mx
from mx_initializers import PReLUInitializer

context = mx.gpu(args.gpu_index)
initializer = PReLUInitializer()

arg_names = loss_group.list_arguments()
arguments = {name : mx.nd.zeros(shape, context) for name, shape in zip(arg_names, arg_shapes)}
for name, array in arguments.items():
  if 'data' in name or 'label' in name: continue
  initializer(name, array)
gradients = {
  name : mx.nd.zeros(shape, context) for name, shape in zip(arg_names, arg_shapes) \
    if not ('data' in name or 'label' in name)
}

state_names = loss_group.list_auxiliary_states()
states = {name : mx.nd.zeros(shape, context) for name, shape in zip(state_names, state_shapes)}
for name, array in states.items(): initializer(name, array)

executor = loss_group.bind(context, arguments, gradients, aux_states=states)

from data_utilities import load_mnist
original = load_mnist(path='stretched_canvas_mnist', scale=1, shape=(1, 56, 56))
stretched = load_mnist(path='stretched_mnist', scale=1, shape=(1, 56, 56))

from mxnet.io import NDArrayIter as Iterator
iterator = Iterator(stretched[0], stretched[1], args.batch_size, shuffle=True)

unpack = lambda batch : (batch.data[0], batch.label[0])

n_iterations = 0
for batch in iterator:
  data, labels = unpack(batch)
  arguments['data'][:] = data
  arguments['labels'][:] = labels
  executor.forward(is_train=True)
  executor.backward()
  for name, gradient in gradients.items():
    arguments[name][:] -= args.lr * gradient
  
  if n_iterations % 100 == 0:
    executor.outputs
