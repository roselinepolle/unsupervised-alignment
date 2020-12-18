#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:16:17 2020

@author: roseline
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
#Custom
import knn
 

#==Version where X_areas,X_grad_areas predefined for each run
#== just do fo each run: 
#X_areas,X_grad_areas = get_system_curve_areas(X,k,quick=True)  
def knn_loss(gf_X,X_areas,X_grad_areas,k,beta,quick = True):
    gfx_areas,gfx_grad_areas = knn.get_system_curve_areas(gf_X,k,quick = quick)
    dist_curves_x_gfx = np.abs(X_areas-gfx_areas) + beta* np.abs(X_grad_areas-gfx_grad_areas)
    loss = np.mean(dist_curves_x_gfx)
    return loss
#Not predefined, ok for quick version.
def knn_loss_calcX(X,gf_X,k,beta,quick = True):
    x_areas,x_grad_areas = knn.get_system_curve_areas(X,k,quick = quick)
    gfx_areas,gfx_grad_areas = knn.get_system_curve_areas(gf_X,k,quick = quick)
    dist_curves_x_gfx = tf.math.abs(x_areas-gfx_areas) + beta* tf.math.abs(x_grad_areas-gfx_grad_areas)
    loss = tf.reduce_mean(dist_curves_x_gfx)
    return loss


def flex_cycle_loss(X_list,cycled_X_list, norm_type = 'l2'):
    """
    Calculates cycle consistency loss for a system and its mapping back
    to itself through the model. Either for two systems simultaneously 
    or for one at a time to allow for performance tr  acking by model component.

    Args:
     - Xs: List of original systems, tensor
     - cycled_Xs: List of resulting system after round-trip. Tensor with same shape as X
     - norm_type: l1 or l2.
    Output:
     - tot_loss: cycle loss per concept
    """
    if norm_type == 'l1':
        dists = [tf.abs(tf.math.subtract(cycled_X_list[i], X_list[i])) for i in range(len(X_list))]
    elif norm_type == 'l2':
        dists = [tf.math.square(tf.math.subtract(cycled_X_list[i], X_list[i])) for i in range(len(X_list))]
    consistencies = [tf.reduce_mean(dists[i]) for i in range(len(X_list))]
    tot_loss = sum(consistencies) / len(consistencies)
    return tot_loss


def distribution_loss(y,fx, gmm_scale):
    """Distribution loss.
    Use upperbound of GMM KL divergence approximation.
    Assumes that incoming `fx` and `y` is all of the data.
    """
    #TODO - Is it not more fiiting fx to gmm_y? + do we need to compare to gmm_y_log_prob?
    #Create gmm from data
    gmm_y = assemble_gmm(y,gmm_scale)
    gmm_fx = assemble_gmm(fx,gmm_scale)
    #Get prob
    gmm_y_log_prob = log_prob_samples_gmm(y,gmm_y)
    gmm_fx_log_prob = log_prob_samples_gmm(y,gmm_fx)

    loss_kl = gmm_y_log_prob - gmm_fx_log_prob
    return loss_kl

def distribution_loss_old(y,fx, gmm_y_samples, gmm_y_log_prob, gmm_scale):
    """Distribution loss.
    Use upperbound of GMM KL divergence approximation.
    Assumes that incoming `fx` and `y` is all of the data.
    """
    
    n_concept = fx.shape[-2]
    weights = [1/n_concept] * n_concept
    gmm_fx = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=fx,
            scale_identity_multiplier=[gmm_scale] * n_concept))
    
    
    #Test override
    gmm_y = assemble_gmm(y,gmm_scale)
    gmm_y_samples = tf.constant(y, dtype=np.float32)
    gmm_y_log_prob = tf.reduce_mean(gmm_y.log_prob(gmm_y_samples), axis=0)
    gmm_fx = assemble_gmm(fx,gmm_scale)
    gmm_fx_log_prob = tf.reduce_mean(gmm_fx.log_prob(gmm_y_samples), axis=0)
    
    loss_kl = gmm_y_log_prob - gmm_fx_log_prob

    return loss_kl

#*******************************************************************
#==================HELPER FUNCTIONS=================================
#*******************************************************************
    
def log_prob_samples_gmm(x,gmm):
    x_samples = x if tf.is_tensor(x) else tf.constant(x, dtype=np.float32)
    log_prob= tf.reduce_mean(gmm.log_prob(x_samples), axis=0)
    return log_prob

def assemble_gmm(x,gmm_scale,batches=False):   
    """Assemble GMM."""
    n_concept = x.shape[-2]
    gmm_weights = [1/n_concept] * n_concept
    gmm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=gmm_weights),
                components_distribution=tfd.MultivariateNormalDiag(
                loc= x ,
                scale_identity_multiplier= [gmm_scale] * n_concept))
    return gmm



def loglik(dist, sample):
    """
    Calculates loglikelihood of drawing a sample from a probability
    distribution

    Args:
     - dist: probability distribution (e.g, output of create_gmm)
    """

    result = tf.math.reduce_mean(dist.log_prob(sample), axis = 1)
    return result





