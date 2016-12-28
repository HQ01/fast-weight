# TODO context GPU

import cPickle as pickle
import sys

import minpy
# minpy.set_global_policy(minpy.OnlyNumPyPolicy)
from minpy.context import set_context, cpu, gpu
set_context(cpu())
# set_context(gpu(0))

from facility import *
from solver_primitives import *
from rnn import FastWeightRNN

INPUT_SIZE = 128
N_HIDDEN = 100
N_CLASSES = 10 # 10 digits
INNER_LENGTH = 3
model = FastWeightRNN(INPUT_SIZE, N_HIDDEN, N_CLASSES, INNER_LENGTH)
initialize(model)
LEARNING_RATE = float(sys.argv[1])
# updater = Updater(model, 'sgd', {'learning_rate' : LEARNING_RATE})
# updater = Updater(model, 'adam', {'learning_rate' : LEARNING_RATE})
updater = Updater(model, 'adam', {'learning_rate' : LEARNING_RATE})

training_X, training_Y = pickle.load(open('../associative_retrieval/training', 'rb'))
validation_X, validation_Y = pickle.load(open('../associative_retrieval/validation', 'rb'))
BATCH_SIZE = 128
X_batches = Batches(training_X, BATCH_SIZE)
Y_batches = Batches(training_Y, BATCH_SIZE)

ITERATIONS = 20000
LOGGING_INTERVAL = 10
VALIDATION_INTERVAL = 50
loss_table = []
validation_accuracy_table = []
sample = lambda N, D : np.random.uniform(-1, 1, (N, D))
for i in range(ITERATIONS):
  X_batch = next(X_batches)
  Y_batch = next(Y_batches)
  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  updater.update(gradients)

  loss = to_float(loss)
  loss_table.append(loss)
  if (i + 1) % LOGGING_INTERVAL == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  if (i + 1) % VALIDATION_INTERVAL == 0:
    predictions = model.forward(validation_X, 'test')
    validation_accuracy = accuracy(predictions, validation_Y)
    print 'iteration %d validation accuracy %f' % (i + 1, validation_accuracy)
    validation_accuracy_table.append(validation_accuracy)

'''
predictions = model.forward(test_X, 'test')
test_accuracy = accuracy(predictions, test_Y)
'''

path = 'retrieval-lr-%f'
info_path = '%s-info' % path
history = (
  test_accuracy,
  loss_table,
  validation_accuracy_table,
)
pickle.dump(history, open(info_path, 'wb'))
