import cPickle as pickle
import numpy as np

n_training_samples = 10000
n_validation_samples = 10000
n_test_samples = 10000
length = 4
dimension = 128

def generate_data(n_samples, length, dimension):
  X = np.zeros((n_samples, length * 2 + 3, dimension), dtype=np.int8)
  character_range = np.arange(ord('A'), ord('Z') + 1, dtype=np.int8)
  characters = np.vstack(
    np.random.choice(character_range, (length,), replace=False) for i in range(n_samples)
  )
  digit_range = np.arange(ord('0'), ord('9') + 1, dtype=np.int8)
  digits = np.vstack(
    np.random.choice(digit_range, (length,), replace=False) for i in range(n_samples)
  )
  for i in range(length):
    X[np.arange(n_samples), i * 2, characters[:, i]] = 1
    X[np.arange(n_samples), i * 2 + 1, digits[:, i]] = 1
  X[np.arange(n_samples), length * 2, ord('?')] = 1
  X[np.arange(n_samples), length * 2 + 1, ord('?')] = 1
  indices = np.random.choice(length, n_samples)
  keys = characters[np.arange(n_samples), indices]
  values = digits[np.arange(n_samples), indices]
  X[np.arange(n_samples), length * 2 + 2, keys] = 1
  Y = values - ord('0')
  return X, Y

def reconstruct(X, Y):
  L, D = X.shape
  X = ''.join(chr(np.argmax(X[i])) for i in range(L))
  Y = chr(int(Y) + ord('0'))
  return X, Y

training_X, training_Y = generate_data(n_training_samples, length, dimension)
validation_X, validation_Y = generate_data(n_validation_samples, length, dimension)
test_X, test_Y = generate_data(n_test_samples, length, dimension)

'''
for i in range(n_training_samples):
  print reconstruct(training_X[i], training_Y[i], dimension)
'''

pickle.dump((training_X, training_Y), open('training', 'wb'))
pickle.dump((validation_X, validation_Y), open('validation', 'wb'))
pickle.dump((test_X, test_Y), open('test', 'wb'))
