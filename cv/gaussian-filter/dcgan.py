import numpy as np

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.context import Context
from mxnet.contrib.autograd import compute_gradient
from mxnet.io import NDArrayIter
from minpy.nn.model_builder import *
from minpy.nn.modules import *

# input shape
gnet_input_size = (100, 1, 1)
nc = 3
ndf = 64
ngf = 64

dnet_input_size=(3, 64, 64)
batch_size = 32
Z = 100
lr = 0.0002
no_bias = True

class Generative(Model):
  def __init__(self, n_filters, n_dfilters, no_bias):
    super(Generative, self).__init__(jit=True)
    self.layers = Sequential (
      self._n_deconvolution(kernel=(4, 4), num_filter=n_filters*8, no_bias=no_bias),
      self._n_deconvolution(kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=n_filters * 4, no_bias=no_bias),
      self._n_deconvolution(kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=n_filters * 2, no_bias=no_bias),
      self._n_deconvolution(kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=n_filters, no_bias=no_bias),
      Deconvolution(kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=n_dfilters, no_bias=no_bias),
      Tanh(),
    )

  @staticmethod
  def _n_deconvolution(**kwargs):
    defaults = {'kernel' : (3, 3), 'no_bias' : True}
    defaults.update(kwargs)
    return Sequential(
      Deconvolution(**defaults),
      BatchNorm(fix_gamma=False),
      ReLU(),
    )

  @Model.decorator
  def forward(self, data):
    data = data
    output = self.layers(data)
    return output

  @Model.decorator
  def loss(self, data, labels):
    return nd.sum(nd.dot(data, labels))

class Discriminative(Model):
  def __init__(self, n_filters, no_bias):
    super(Discriminative, self).__init__(jit=True)
    kwargs = {'kernel': (4, 4), 'stride': (2, 2), 'pad': (1, 1), 'no_bias': no_bias}
    self.layers = Sequential(
      self._convolution(num_filter=n_filters, **kwargs),
      self._n_convolution(num_filter=n_filters * 2, **kwargs),
      self._n_convolution(num_filter=n_filters * 4, **kwargs),
      self._n_convolution(num_filter=n_filters * 8, **kwargs),
      Convolution(kernel=(4, 4), num_filter=1, no_bias=no_bias),
      BatchFlatten(),
    )

  @staticmethod
  def _n_convolution(**kwargs):
    defaults = {'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'no_bias': True}
    defaults.update(kwargs)
    return Sequential(
      Convolution(**defaults),
      BatchNorm(fix_gamma=False),
      LeakyReLU(slope=0.2),
    )

  @staticmethod
  def _convolution(**kwargs):
    defaults = {'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'no_bias': True}
    defaults.update(kwargs)
    return Sequential(
      Convolution(**defaults),
      LeakyReLU(slope=0.2),
    )

  @Model.decorator
  def forward(self, data):
    return self.layers(data)

  @Model.decorator
  def loss(self, data, labels):
    print labels.asnumpy()
    return nd.LogisticRegressionOutput(data, labels)

class RandIter(mx.io.DataIter):
  def __init__(self, batch_size, D):
    self.batch_size = batch_size
    self.D = D
    self.provide_data = [('rand', (batch_size, D, 1, 1))]
    self.provide_label = [np.zeros(batch_size)]

  def iter_next(self):
    return True

  def getdata(self):
    return [nd.normal(0, 1.0, shape=(self.batch_size, self.D, 1, 1)).copyto(Context.default_ctx)]

def fetch_and_get_mnist():
  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original')

  '''
  np.random.seed(1234)
  p = np.random.permutation(mnist.data.shape[0])
  X = mnist.data[p]
  X = X.reshape((70000, 28, 28))
  '''
  X = mnist.data[:70]
  X = X.reshape((70, 28, 28))

  from scipy.misc import imresize
  X = np.asarray([imresize(x, (64,64)) for x in X])
  X = X.astype(np.float32) / (255.0 / 2) - 1.0
  X = X.reshape((70, 1, 64, 64))
  X = np.tile(X, (1, 3, 1, 1))
  X_train = X[:60]
  X_test = X[60:]

  return X_train, X_test

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_index', type=int, default=0)
  parser.add_argument('--path', type=str, default='standard-scale-mnist')
  args = parser.parse_args()

  context = mx.cpu() if args.gpu_index < 0 else mx.gpu(args.gpu_index)
  Context.default_ctx = context

  # Create model.
  gnet_model = Generative(ngf, nc, no_bias)
  dnetwork = Discriminative(ndf, no_bias)
  
  # Prepare data
  '''
  from data_utilities import load_mnist
  X_training, _, X_test, _, _, _ = load_mnist(path=args.path, normalize=True, shape=(1, 56, 56))
  '''
  X_training, X_test = fetch_and_get_mnist()
  real_data = NDArrayIter(X_training, np.ones(X_training.shape[0]), batch_size=batch_size)
  random_data = RandIter(batch_size, Z)

  gnet_updater = Updater(gnet_model, update_rule='sgd_momentum', lr=0.1, momentum=0.9)
  dnet_updater = Updater(dnetwork, update_rule='sgd_momentum', lr=0.1, momentum=0.9)

  # Training    
  epoch_number = 0
  iteration_number = 0
  terminated = False

  while not terminated:
    epoch_number += 1
    real_data.reset()
    i = 0 
    for real_batch in real_data:
      random_batch = random_data.getdata()
      dnet_real_output = dnetwork.forward(real_batch.data[0], is_train=True)
      dnet_real_loss = dnetwork.loss(dnet_real_output, real_batch.label[0], is_train=True)
      print dnet_real_loss.context
      compute_gradient((dnet_real_loss,))
      dnet_real_grad_dict = dnetwork.grad_dict

      '''
      copy = lambda array: array.copyto(mx.cpu())
      cache_dict = dict(zip(dnetwork.grad_dict.keys(), map(copy, dnetwork.grad_dict.values())))
      '''

      generated_data = gnet_model.forward(random_batch[0], is_train=True)
      dnet_fake_output = dnetwork.forward(generated_data, is_train=True)
      dnet_fake_loss = dnetwork.loss(dnet_fake_output, nd.zeros(generated_data.shape[0]), is_train=True)
      compute_gradient((dnet_fake_loss,))

      '''
      for key, value in dnetwork.grad_dict.items():
        value += cache_dict[key].copyto(value.context)
      '''

      for each_key in dnet_real_grad_dict:
          dnetwork.grad_dict[each_key] += dnet_real_grad_dict[each_key]
      dnet_updater(dnetwork.grad_dict)

      # ff dnet using fake data and real label
      generated_data = gnet_model.forward(random_batch[0], is_train=True)
      dnet_fake_output = dnetwork.forward(generated_data, is_train=True)
      dnet_loss_to_train_gnet = dnetwork.loss(dnet_fake_output, nd.ones(generated_data.shape[0]), is_train=True)
      compute_gradient((dnet_loss_to_train_gnet,))
      gnet_updater(gnet_model.grad_dict)
      print dnet_loss_to_train_gnet.asnumpy()
