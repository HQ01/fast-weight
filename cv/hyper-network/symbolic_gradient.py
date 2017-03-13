import mxnet as mx
import numpy as np
import mx_layers as layers
from mx_utility import output_shape, mxnet_array_mapping

def _break_graph(operations, name, data_shape):
  network = layers.variable('data')
  for operation in operations:
    if network.name == name:
      replaced = network
      shape = output_shape(replaced, data=data_shape)
      network = operation(layers.variable('%s_data' % name, shape=shape))
    else:
      network = operation(network)
  return replaced, network

def _filter_arguments(symbol, parameters, states):
  return \
    {key : value for key, value in parameters.items() if key in symbol.list_arguments()}, \
    {key : value for key, value in states.items() if key in symbol.list_auxiliary_states()},

def symbolic_gradient(operations, X, data, labels, parameters, states, context=None):
  data_shape = data.shape
  X_symbol, loss_symbol = _break_graph(operations, X, data_shape)
  parameters = mxnet_array_mapping(parameters, context)
  states = mxnet_array_mapping(states, context)
  X_parameters, X_states = _filter_arguments(X_symbol, parameters, states)
  X_parameters['data'] = mx.nd.array(data, context)
  if context is None: context = mx.cpu()
  X_executor = X_symbol.bind(context, X_parameters, grad_req='null', aux_states=X_states)
  X_value = X_executor.forward()[0]
  loss_parameters, loss_states = _filter_arguments(loss_symbol, parameters, states)
  loss_parameters['%s_data' % X] = X_value
  loss_parameters['softmax_label'] = mx.nd.array(labels, context)
  gradient = {'%s_data' % X : mx.nd.zeros(X_value.shape, context)}
  loss_executor = loss_symbol.bind(context, loss_parameters, args_grad=gradient, aux_states=loss_states)
  loss_executor.forward()
  loss_executor.backward()
  return gradient['%s_data' % X].asnumpy()
