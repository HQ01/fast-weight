import cPickle as pickle
import numpy as np
import sys

N = int(sys.argv[1])
T, _ = pickle.load(open('parameters/triple-state-transitory-residual-network', 'rb'))
R, _ = pickle.load(open('parameters/triple-state-refined-residual-network-%d' % N, 'rb'))
for key, value in T.items():
  print key, np.max(np.abs(value - R[key]))
