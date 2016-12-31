import mxnet.symbol as symbol

# class WeightSharingResidualNetwork:
def weight_sharing_residual_network(graph):
  X = symbol.Variable('data')
  for index, node in enumerate(graph):
    weight = symbol.Variable('convolution_weight_%d' % index)
    bias = symbol.Variable('convolution_bias_%d' % index)
    kwargs, activation, times = node
    for t in range(times):
      X = symbol.Convolution(data = X, weight = weight, bias = bias, **kwargs)
      
