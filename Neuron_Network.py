import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model3(num_layers, width)
        self._num_actions = 3

    def _build_model3(self, num_layers, width):
        """
        Build and compile with CNN
        """
        input_pos = keras.Input(shape=(self._input_dim, ), name="pos")

        x = layers.Dense(256, kernel_initializer='uniform', activation='relu')(input_pos)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, kernel_initializer='uniform', activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        action = layers.Dense(self._num_actions, activation="softmax")(x) # the probability of the action space
        critic = layers.Dense(1)(x)                                       # the estimate reward for the input state

        model = keras.Model(inputs=input_pos, outputs=[action, critic], name='DNN_model')
        model.compile(loss=losses.mean_squared_error, optimizer=RMSprop(lr=self._learning_rate))

        keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

        # return [action_prob, estimate reward]
        return model
