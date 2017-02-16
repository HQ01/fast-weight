from stochastic_fully_connected_network import StochasticFullyConnectedNetwork
from minpy.nn.layers import relu
from solver_primitives import initialize

model = StochasticFullyConnectedNetwork(3072, 10, 3, relu, 0.5)
initialize(model)
for key, value in model.params.items():
  print key, value.shape
