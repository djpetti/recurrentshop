from keras.models import Model
import keras.layers as layers
import keras.optimizers as optimizers

import numpy as np

from recurrentshop import LSTMCell, RecurrentModel


""" Example: Create a simple LSTM that supports readout with an initial readout
input. """

# Main cell input.
cell_input = layers.Input(shape=(10,))
# Readout input.
readout_input = layers.Input(shape=(100,))

# Internal inputs for the LSTM cell.
last_state = layers.Input(shape=(100,))
last_output = layers.Input(shape=(100,))

# Create the LSTM layer.
fused_inputs = layers.concatenate([cell_input, readout_input])
lstm1_o, lstm1_h, lstm1_c = LSTMCell(100)([cell_input, last_state, last_output])

# Build the RNN.
rnn = RecurrentModel(input=cell_input,
                     output=lstm1_o,
                     initial_states=[last_state, last_output],
                     final_states=[lstm1_h, lstm1_c],
                     readout_input=readout_input)

# Main sequence input.
sequence_input = layers.Input(shape=(50, 10))
# Initial readout input.
initial_readout = layers.Input(shape=(100,))

rnn_output = rnn(sequence_input, initial_readout=initial_readout)

# Build the Keras model.
model = Model(inputs=[sequence_input, initial_readout], outputs=rnn_output)
opt = optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss="mean_squared_error", optimizer=opt)

# Train the model on some random data.
x = np.random.rand(128, 50, 10)
y = np.random.rand(128, 100)
initial_readout = np.random.rand(128, 100)

model.fit([x, initial_readout], y)
