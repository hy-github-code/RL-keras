import torch
import torch.nn as nn
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import *
import tensorflow as tf
import numpy as np


class DQN_keras():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.net_work()
        self.model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')
        self.model.summary()
        self.tgt_model = self.net_work()
        self.tgt_model.compile(optimizer='adam', loss='mse')

    def net_work(self):
        inp = Input(self.state_dim)
        temp = Conv2D(32, kernel_size=8, strides=4, activation='relu', data_format='channels_first')(inp)
        temp = Conv2D(64, kernel_size=4, strides=2, activation='relu', data_format='channels_first')(temp)
        temp = Conv2D(64, kernel_size=3, strides=1, activation='relu', data_format='channels_first')(temp)
        temp = Flatten()(temp)
        result = Dense(self.action_dim)(temp)
        return Model(inp, result)
