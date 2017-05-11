import mxnet.ndarray as nd
from minpy.nn.model_builder import *
from minpy.nn.modules import *

class MSP(Layer):
  _module_name = 'msp'

  def __init__(self, n_scales):
    super(MSP, self).__init__(params=('weight',))
    self._n_scales = n_scales
#   init_configs = {'weight' : {'init_rule' : 'constant', 'value' : nd.array([0.1, 0.9, 0.1]).reshape((1, n_scales, 1, 1, 1))}}
    init_configs = {'weight' : {'init_rule' : 'constant', 'value' : 1.0 / self._n_scales}}
    self._register_init_configs(init_configs)

  def forward(self, X):
    filters = []
    for i in range(self._n_scales):
      kernel = (i * 2 + 1,) * 2
      pad = (i,) * 2
      f = nd.Pooling(data=data, pool_type='max', kernel=kernel, stride=(1, 1), pad=pad)
      f = nd.reshape(f, (f.shape[0], 1) + f.shape[1:])
      filters.append(f)

    filters = nd.concat(*filters, dim=1)
    weight = nd.softmax(self._get_param(self.weight), axis=1)
#   weight = self._get_param(self.weight)
    filters = nd.sum(filters * weight, axis=1)

    return filters
 
  def param_shapes(self, shape):
    return {self.weight : (1, self._n_scales, 1, 1, 1)}

class MSPCNN(Model):
  def __init__(self, n_filters, n_scales, n_units):
    super(MSPCNN, self).__init__(jit=True)

    self._msp = MSP(n_scales)
    self._convolutions = (
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
      Convolution(num_filter=n_filters, kernel=(5, 5), stride=(1, 1), pad=(2, 2)),
    )
    self._linear = FullyConnected(num_hidden=n_units)
    self._classifier = FullyConnected(num_hidden=10)

  @Model.decorator
  def forward(self, data):
    data = self._msp(data)
    for i, c in enumerate(self._convolutions):
      data = c(data)
      data = Tanh()(data)
      data = nd.Pooling(data=data, pool_type='max', kernel=(2, 2), stride=(2, 2), pad=(0, 0))

    data = nd.max(data, axis=1)
    data = self._linear(data)
    data = Tanh()(data)
    data = self._classifier(data)

    return data

  @Model.decorator
  def loss(self, data, labels):
    return nd.SoftmaxOutput(data, labels, normalization='batch')

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--path', type=str, default='standard-scale-mnist')
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--gpu_index', type=int, default=0)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--n_epochs', type=int, default=25)
  parser.add_argument('--n_filters', type=int, default=4)
  parser.add_argument('--n_scales', type=int, default=3)
  parser.add_argument('--n_units', type=int, default=16)
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

  model = MSPCNN(args.n_filters, args.n_scales, args.n_units)
  updater = Updater(model, update_rule='adam', lr=args.lr)
# updater = Updater(model, update_rule='sgd_momentum', lr=1e-1, momentum=0.9)
  
  import numpy as np
  from mxnet.contrib.autograd import compute_gradient
  import minpy.nn.utils as utils
  
  validation_accuracy = []

  for epoch in range(args.n_epochs):
    training_data.reset()
    for iteration, batch in enumerate(training_data):
      data, labels = unpack_batch(batch)
      predictions = model.forward(data, is_train=True)
      loss = model.loss(predictions, labels, is_train=True)
      compute_gradient((loss,))
      '''
      for key, value in model.params.items():
        print key, nd.max(value).asnumpy(), nd.min(nd.abs(value)).asnumpy()
      '''
      updater(model.grad_dict)

    loss_value = utils.cross_entropy(loss, labels)
    print 'epoch %d loss %f' % (epoch, loss_value)

    if epoch < 1:
      print utils.count_params(model.params)

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

    if epoch > 0 and validation_accuracy[-1] < min(validation_accuracy[:-1]):
      params = utils.copy_arrays(model.params)

  model.params = utils.copy_arrays(params, context)
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

identifier = 'msp-cnn-%d-filters-%d-units-%s' % (args.n_filters, args.n_units, args.path)
history = validation_accuracy, test_accuracy
from joblib import dump
dump(history, 'info/%s' % identifier)
dump(params, 'parameters/_')
