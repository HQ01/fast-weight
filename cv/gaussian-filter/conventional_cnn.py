import mxnet.ndarray as nd
from minpy.nn.model_builder import *
from minpy.nn.modules import *

class ConventionalCNN(Model):
  def __init__(self, n_filters):
    super(ConventionalCNN, self).__init__(jit=False)
    self._convolutions = (
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
    )
    self._c0 = Convolution(num_filter=n_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    self._linear = FullyConnected(num_hidden=10)

  @Model.decorator
  def forward(self, data):
    for c in self._convolutions:
      data = c(data)
      data = ReLU()(data)
      data = nd.Pooling(data=data, pool_type='max', kernel=(2, 2), stride=(2, 2), pad=(0, 0))

    data = self._c0(data)
    data = ReLU()(data)

    data = nd.Pooling(data=data, pool_type='avg', kernel=(7, 7), stride=(1, 1), pad=(0, 0))
    data = nd.flatten(data)
    data = self._linear(data)

    return data

  @Model.decorator
  def loss(self, data, labels):
    return nd.SoftmaxOutput(data, labels, normalization='batch')

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--path', type=str, default='multi-scale-mnist')
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--gpu_index', type=int, default=0)
  parser.add_argument('--n_epochs', type=int, default=25)
  parser.add_argument('--n_filters', type=int, default=16)
  args = parser.parse_args()

  import mxnet as mx
  from mxnet.context import Context
  context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
  Context.default_ctx = context

  unpack_batch = lambda batch : \
      (batch.data[0].as_in_context(context), batch.label[0].as_in_context(context))

  from data_utilities import load_mnist
  data = load_mnist(path=args.path, normalize=True, shape=(1, 56, 56))

  from mxnet.io import NDArrayIter
  training_data = NDArrayIter(data[0], data[1], batch_size=args.batch_size)
  validation_data = NDArrayIter(data[2], data[3], batch_size=args.batch_size)
  test_data = NDArrayIter(data[4], data[5], batch_size=args.batch_size)

  model = ConventionalCNN(args.n_filters)
  updater = Updater(model, update_rule='adam', lr=1e-3)
# updater = Updater(model, update_rule='sgd_momentum', lr=1e-1, momentum=0.9)
  
  import numpy as np
  from mxnet.contrib.autograd import compute_gradient
  from minpy.nn.utils import cross_entropy
  
  validation_accuracy = []

  for epoch in range(args.n_epochs):
    training_data.reset()
    for iteration, batch in enumerate(training_data):
      data, labels = unpack_batch(batch)
      predictions = model.forward(data, is_train=True)
      loss = model.loss(predictions, labels, is_train=True)
      compute_gradient((loss,))
      updater(model.grad_dict)

    loss_value = cross_entropy(loss, labels)
    print 'epoch %d loss %f' % (epoch, loss_value)

    validation_data.reset()
    n_errors, n_samples = 0, 0
    for batch in validation_data:
      data, labels = unpack_batch(batch)
      scores = model.forward(data)
      predictions = nd.argmax(scores, axis=1)
      errors = (predictions - labels).asnumpy()
      n_errors += np.count_nonzero(errors)
      n_samples += data.shape[0]

    validation_accuracy.append(n_errors / float(n_samples))
    print 'epoch %d validation error %f' % (epoch, validation_accuracy[-1])

  test_data.reset()
  n_errors, n_samples = 0, 0
  for batch in test_data:
    data, labels = unpack_batch(batch)
    scores = model.forward(data)
    predictions = nd.argmax(scores, axis=1)
    errors = (predictions - labels).asnumpy()
    n_errors += np.count_nonzero(errors)
    n_samples += data.shape[0]

  test_accuracy = n_errors / float(n_samples)
  print 'test error %f' % test_accuracy

identifier = 'conventional-cnn-%d-filters-%s' % (args.n_filters, args.path)
history = validation_accuracy, test_accuracy
from joblib import dump
dump(history, 'info/%s' % identifier)
