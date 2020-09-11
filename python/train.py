#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:24:13 2020

@author: roseline
"""

import losses
import utils
import tensorflow as tf

def train_batch_step(model_f,model_g,optimizer,x_batch, y_batch, loss_cycle_scale):
        with tf.GradientTape(persistent=True) as tape_cycle:
            x_c = model_g(model_f(x_batch))
            y_c = model_f(model_g(y_batch))
            cycle_loss = loss_cycle_scale * losses.flex_cycle_loss([x_batch,y_batch],[x_c,y_c], norm_type = 'l2')
            #cycle_loss = losses.cycle_loss_func(x_batch, y_batch, x_c, y_c, loss_cycle_scale)
        gradients_cycle_f = tape_cycle.gradient(cycle_loss, model_f.trainable_variables)
        gradients_cycle_g = tape_cycle.gradient(cycle_loss, model_g.trainable_variables)
        del tape_cycle
        optimizer.apply_gradients(zip(gradients_cycle_f, model_f.trainable_variables))
        optimizer.apply_gradients(zip(gradients_cycle_g, model_g.trainable_variables))
        return cycle_loss
    
def train_full_step(model_f,model_g,optimizer,x_all, y_all, loss_distr_scale,gmm_scale):
        with tf.GradientTape() as tape_f0:
            f_x = model_f(x_all)
            dist_loss_f = loss_distr_scale * losses.distribution_loss(y_all, f_x, gmm_scale)
        gradients_target_fx = tape_f0.gradient(dist_loss_f, model_f.trainable_variables)
        with tf.GradientTape() as tape_g0:
            g_y = model_g(y_all)
            dist_loss_g = loss_distr_scale * losses.distribution_loss(x_all, g_y, gmm_scale)
        gradients_target_gy = tape_g0.gradient(dist_loss_g, model_g.trainable_variables)
        # Combine gradients.
        optimizer.apply_gradients(zip(gradients_target_fx, model_f.trainable_variables))
        optimizer.apply_gradients(zip(gradients_target_gy, model_g.trainable_variables))
        return dist_loss_f, dist_loss_g

    