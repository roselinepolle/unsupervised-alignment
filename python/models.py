#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:10:26 2020

@author: roseline
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import numpy as np
import utils


#==============
class MLP(Model):
    """Model that aligns two domains."""
    def __init__(self, n_dim_0, n_dim_1):
        """Initialize."""
        n_hidden = 100
        # NOTE: elu sometimes works
        super(MLP, self).__init__()
        self.d1 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        self.d2 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        self.d3 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        self.d5 = Dense(n_dim_1, activation='tanh')

    def call(self, x):
        """Call."""
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)  # TODO was off, turned on for 5D
        x4 = self.d5(x3)
        return x4
    
    
class Encoder(Model):
    """ Encoder class to shared latent space"""
    def __init__(self, hidden, n_dim_encode):
        super(Encoder, self).__init__()
        self.d1 = Dense(hidden, activation='tanh')
        self.d2 = Dense(hidden, activation='tanh')
        self.d3 = Dense(hidden, activation='tanh')
        self.d4 = Dense(n_dim_encode, activation='linear')


    def call(self, x):
        """Feed-forward pass."""
        h1 = self.d1(x)
        h2 = self.d2(h1)
        h3 = self.d3(h2)
        y = self.d4(h3)
        return y


class Decoder(Model):
    """ Decoder block from shared latent space"""
    def __init__(self, hidden, n_dim_out):
        super(Decoder, self).__init__()
        self.d1 = Dense(hidden, activation='tanh')
        self.d2 = Dense(hidden, activation='tanh')
        self.d3 = Dense(hidden, activation='tanh')
        self.d4 = Dense(n_dim_out, activation='linear')
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
    
#####===============HELPER FUNCTIONS    
def init_model(cond_best,thresh,restart,sd_restart,emb_dim,best_model):
    #if cond_best_model = True => best model. Else:
    #if below thresh=> best weight, else new model
    #bw = best weights
    new = False
    if cond_best or (np.random.uniform() < thresh and restart > 0):
        model = best_model
    else:
        utils.set_seed_np_tf(sd_restart)
        model = MLP(emb_dim, emb_dim)
        new = True
    return model, new
    
