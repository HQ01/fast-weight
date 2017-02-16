from minpy.nn.layers import affine, relu
from minpy.nn.model import ModelBase

class StochasticFullyConnectedNetwork(ModelBase):
  def __init__(self, refining_times, dimension, n_classes, nonlinear, decay_rate):
    super(StochasticFullyConnectedNetwork, self).__init__()
    self._refining_times = refining_times
    self._decay_rate = decay_rate
    self._nonlinear = nonlinear
    self._n_classes = n_classes
    self._dimensions = (dimension, dimension / 2, dimension / 4, dimension / 8, n_classes)
    for index, d in enumerate(zip(self._dimensions[:-1], self._dimensions[1:])):
      d_from, d_to = d
      self \
        .add_param(name='W%d' % index, shape=(d_from, d_to), init_rule='xavier') \
        .add_param(name='b%d' % index, shape=(d_to,), init_rule='constant', init_config={'value' : 0}) \
        .add_param(name='shared_W%d' % index, shape=(d_to, d_to), init_rule='xavier') \
        .add_param(name='shared_b%d' % index, shape=(d_to,), init_rule='constant', init_config={'value' : 0})

  def forward(self, X, mode):
    for index in range(len(self._dimensions) - 2):
      W, b = self.params['W%d' % index], self.params['b%d' % index]
      network = affine(network, W, b)
      network = self._nonlinear(network)
      shared_W, shared_b = self.params['shared_W%d' % index], self.params['shared_b%d' % index]
      for t in range(self._refining_times):
        residual = affine(network, shared_W, shared_b)
        residual = self._nonlinear(network)
        network = self._decay_rate * network + (1 - self._decay_rate) * residual
    W, b = self.params['W%d' % (len(self._dimensions) - 1)], self.params['b%d' % (len(self._dimensions) - 1)]
    network = affine(network, W, b)
    return network

  def loss(self, predictions, Y):
    return softmax_loss(predictions, Y)

if __name__ is '__main__':
  network = StochasticFullyConnectedNetwork(3072, 10, 3, relu, 0.5)
  print network.params.keys()
