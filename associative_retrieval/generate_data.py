import cPickle as pickle
import numpy as np

n_training_samples = 1
n_validation_samples = 1
n_test_samples = 1
length = 4
dimension = 128

def generate_data(n_samples, length, dimension):
  X = np.zeros((n_samples, length * 2 + 3, dimension), dtype=np.int)
  characters = np.random.choice(np.arange(ord('A'), ord('Z') + 1), (n_samples, length), replace=False)
  digits = np.random.choice(np.arange(ord('0'), ord('9') + 1), (n_samples, length), replace=False)
  for i in range(length):
    X[np.arange(n_samples), i * 2, characters[:, i]] = 1
    X[np.arange(n_samples), i * 2 + 1, digits[:, i]] = 1
  X[np.arange(n_samples), length * 2, ord('?')] = 1
  X[np.arange(n_samples), length * 2 + 1, ord('?')] = 1
  indices = np.random.choice(length, n_samples)
  keys = characters[:, indices]
  values = digits[:, indices]
  X[np.arange(n_samples), length * 2 + 2, keys] = 1
  Y = values
  return X, Y

def reconstruct(X, Y, dimension):
  L, D = X.shape
  X = ''.join(chr(np.argmax(X[i])) for i in range(L))
  Y = chr(Y)
  return X, Y

training_X, training_Y = generate_data(n_training_samples, length, dimension)
print training_X.shape, training_Y.shape
validation_X, validation_Y = generate_data(n_validation_samples, length, dimension)
test_X, test_Y = generate_data(n_test_samples, length, dimension)

pickle.dump((training_X, training_Y), open('training', 'wb'))
pickle.dump((validation_X, validation_Y), open('validation', 'wb'))
pickle.dump((test_X, test_Y), open('test', 'wb'))
