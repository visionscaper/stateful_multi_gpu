"""
Trains a multi GPU version of a stateful Keras model and a single GPU version
"""
import time

import tensorflow as tf

from keras import backend as K

from keras.layers import Input
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from keras.models import Model

# The original multi_gpu_model util of Keras
#from keras.utils.training_utils import multi_gpu_model

import numpy as np

from util import stateful_multi_gpu

###################### SETUP #######################
num_gpus = 2
training_batch_size = 128

seq_len = 30
num_symbols = 50
num_classes = 10
RNNLayer = GRU
state_size = 256
stateful_model = True

batch_training_steps = 50
reset_period = 2  # reset model every two training steps

config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
K.set_session(sess)
#####################################################


########## CREATE STATEFUL MULTI GPU MODEL #########
def inputs_generator(batch_size):
    print('Creating inputs for batch_size %d' % batch_size)

    rnn_input = Input(
        name="rnn-input-%d" % batch_size,
        batch_shape=(batch_size, seq_len, num_symbols))

    return rnn_input


def model_generator(batch_size):
    inputs = inputs_generator(batch_size)

    layer_output = RNNLayer(
        state_size,
        stateful=stateful_model,
        return_sequences=True)(inputs)
    outputs = TimeDistributed(Dense(num_classes))(layer_output)

    return Model(inputs=inputs, outputs=outputs)

print()
print('Creating multi GPU stateful model')
parallel_model = stateful_multi_gpu(inputs_generator, model_generator, training_batch_size, num_gpus)

parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

parallel_model.summary()
####################################################


########## CREATE STATEFUL SINGLE GPU MODEL #########
print()
print('Creating single GPU stateful model')
with tf.device('/gpu:0'):
    with tf.name_scope('single_gpu_model'):
        single_gpu_model = model_generator(training_batch_size)
        single_gpu_model.compile(loss='categorical_crossentropy',
                                 optimizer='rmsprop')

        single_gpu_model.summary()
####################################################


################# TRAINING TESTS ###################
def train(model):
    for step in range(batch_training_steps):

        start = time.time()

        if step % reset_period:
            print("Reset states")
            parallel_model.reset_states()

        input_batch = np.random.rand(training_batch_size, seq_len, num_symbols)
        target_batch = np.random.rand(training_batch_size, seq_len, num_classes)

        metrics = model.train_on_batch(input_batch, target_batch)

        end = time.time()
        print("Step %d : loss : %0.3f : time : %0.3f" % (step, metrics, end - start))


print()
print('Training multi GPU stateful model')
train(parallel_model)

print()
print('Training single GPU stateful model')
train(single_gpu_model)
####################################################
