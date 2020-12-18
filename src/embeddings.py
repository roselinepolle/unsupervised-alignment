# -*- coding: utf-8 -*-
# Copyright 2020 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module managing embedding creation and loading."""

from pathlib import Path
import numpy as np
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import scipy as sp
from scipy import stats
import math
import numpy as np
from tensorflow.keras import backend as K
#Custom
import utils

####====================================================
#####=====================RP============================
####====================================================

def get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=True):
    #Get systems of embeddings (as np arrays, float32)
    systems, noisy_Xs = get_N_systems(n_systems = n_systems, noise_size = noise,rotation=True,plot=bplot, 
                                      num_concepts=n_concepts,n_dim=emb_dim,n_epicentres =n_epicentres,linearsep =linearsep,seed = sd)
    X_A, X_B = noisy_Xs
    A, B = systems
    #Shuffle data
    A_shuff,A_shuff_idx, A_idx_map = utils.shuffle_system(A)
    B_shuff,B_shuff_idx, B_idx_map = utils.shuffle_system(B)
    # Verify mapping to be safe.
    np.testing.assert_array_equal(A, A_shuff[A_idx_map,:])
    np.testing.assert_array_equal(B, B_shuff[B_idx_map,:])
    return A,B,X_A,X_B,B_shuff,B_idx_map



def create_clumpy_system(num_concepts=200,n_dim=2,n_epicentres =1,linearsep = 1,plot=False,seed = 456):
    """
    Create a single clumpy gaussian system (possibily composed of several generating gaussians, i.e. mixture of gaussians). 

    Parameters
    ----------
    num_concepts : Scalar. Number of concepts. The default is 200.
    n_dim : Scalar. Dimension of each embedding. The default is 2.
    n_epicentres : Scalar. Number of gaussians generating the embeddings. The default is 1.
    linearsep : From 1 to 10. Reduces the variance of the generating Gaussians in 1/linearsep**4. From 1 to 10. Default is 1. 
    plot : Boolean. Plot the concepts if n_dim==2D and if set to True .The default is False.

    Returns
    -------
    X: Numpy array of shape (n_concepts,n_dim). Stores the resulting embeddings. 
    """
    
    #Settings
    np.random.seed(seed)
    epicentre_range=1 
    
    #Check number of epicentres allows 40 items per epicentre
    min_concepts_per_epicentres = 40
    max_epicentres = num_concepts // min_concepts_per_epicentres
    assert (n_epicentres <= max_epicentres),"Max number of epicentres is %d for %d concpets" %(max_epicentres,num_concepts)
    
    #Get sigma
    #sigma = 1/linearsep
    #sigma = 1/math.exp(linearsep)
    sigma = 1/linearsep**3
    
    # Create X (from create_n_systems)
    X_cov = np.zeros((n_dim, n_dim), dtype = np.float32)
    np.fill_diagonal(X_cov, sigma)
    
    means = np.random.uniform(
        -epicentre_range, epicentre_range, size = (n_epicentres, n_dim)
        )

    X = []
    for i in range(num_concepts):
        mean = i % n_epicentres
        value = np.random.multivariate_normal(
            mean=means[mean], cov=X_cov,  size=1
            )
        X.append(value)
    X = np.array(X,dtype = np.float32)
    X = np.squeeze(X)
    
    if n_dim == 2 and plot == True:
        fig = plt.figure()
        plt.scatter(
         X[:,0], X[:,1]
         ) 
        plt.title("Synthetic data")
        plt.show()
    
    X = np.float32(X)
    return X
    

def get_N_systems(n_systems = 2, noise_size = 0.1,rotation=True,plot=False, 
                  num_concepts=200,n_dim=2,n_epicentres =1,linearsep = 1,seed = 456):
    
    """
    Create N clumpy gaussian systems. The firt one is created using 'create_clumpy_system',
    the others by adding a random noise and random rotation

    Parameters
    ----------
    n_systems: Scalar. Number of systems to create. Default is 2. 
    noise_size: Scalar. Stdev of noise to add between systems. 
    rotation: Boolean. If False, no rotation is done between systems. 
    plot : Boolean. Plot the concepts if n_dim==2D and n_systems<=3 and if set to True .The default is False.
    num_concepts : Scalar. Number of concepts. The default is 200.
    n_dim : Scalar. Dimension of each embedding. The default is 2.
    n_epicentres : Scalar. Number of gaussians generating the embeddings. The default is 1.
    linearsep : From 1 to 10. Reduces the variance of the generating Gaussians in 1/linearsep**4. From 1 to 10. Default is 1. 
    

    Returns
    -------
    systems : List of numpy array of shape (n_concepts,n_dim). Each list item stores the embeddings of one system. 
    noisy_Xs : List of numpy array of shape (n_concepts,n_dim). Stored before rotation but after noise. 
    """
    
    
    #Settings
    np.random.seed(seed)
    
    #Create base X (numpy array) and store as first system
    X = create_clumpy_system(num_concepts,n_dim,n_epicentres,linearsep,plot=False,seed = seed)
    X = utils.preprocess_embedding(X)
    X = np.float32(X)
    systems = [X]
    noisy_Xs = [X]  
    

    #Create n_systems as rotation
    for i in range(n_systems-1):
        #Add noise to X and store
        noisy_X = X + np.random.normal(loc=0,scale =noise_size ,size = X.shape) #noise_size = variance
        noisy_X = utils.preprocess_embedding(noisy_X)
        noisy_X = np.float32(noisy_X)
        noisy_Xs.append(noisy_X)
        if rotation == True:
            # Create Y by rotating noisy X
            random_rot_mat = sp.stats.special_ortho_group.rvs(n_dim)
            Y = np.matmul(random_rot_mat, noisy_X.T)
            Y = Y.T
        else:
            Y = noisy_X
        Y = np.float32(Y)
        systems.append(Y)
             
    #Plot
    if plot == True and n_systems <= 3:
        if n_dim == 2:
            fig, axes = plt.subplots(1, 2)
            axes[0].set_title('Noise only')
            axes[1].set_title('Noise + rotation')
            colors = ["#f64b4bff","#4b64f6ff",'g']
            labels = ["System A","System B","System C"]       
            for i in range(n_systems): 
                axes[0].scatter(noisy_Xs[i][:,0], noisy_Xs[i][:,1],color = colors[i], label = labels[i])
                axes[1].scatter(systems[i][:,0], systems[i][:,1],color = colors[i], label = labels[i]) 
            fig.suptitle("Synthetic data")
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            plt.show()
        elif n_dim == 3:
            fig, axes = plt.subplots(figsize=(10., 5.))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax1.set_title('Noise only')
            ax2.set_title('Noise + rotation')
            colors = ["#f64b4bff","#4b64f6ff",'g']
            labels = ["System A","System B","System C"]       
            for i in range(n_systems): 
                ax1.scatter(noisy_Xs[i][:,0], noisy_Xs[i][:,1],color = colors[i], label = labels[i])
                ax2.scatter(systems[i][:,0], systems[i][:,1],color = colors[i], label = labels[i]) 
            fig.suptitle("Synthetic data")
            handles, labels = ax2.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            plt.show()  
        
    #For tensor output
    # systems = [tf.expand_dims(tf.convert_to_tensor(sys, type = K.floatx()),0) for sys in systems]
    # noisy_Xs = [tf.expand_dims(tf.convert_to_tensor(nois, dtype = K.floatx()),0) for nois in noisy_Xs]
    
    return systems, noisy_Xs
        
        

####========================================================
#####=====================BEFORE============================
####========================================================
def noisy_gaussian(n_concept, noise=0, n_dim=2, seed=7849):
    """Load synthetic embeddings drawn from a multivariate Gaussian.

    Arguments:
        n_concept: A scalar indicating the number of concepts.
        noise (optional): A scalar indicating the amount of noise to
            add. This should be between 0 and 1.
        n_dim (optional): A scalar indicating the number of dimensions.
        seed (optional): A scalar indicating the seed to use in the
            random number generator.

    Returns:
        z_0: The first embedding.
        z_1: The second embedding.

    """
    # Create synthetic embeddings.
    np.random.seed(seed)
    z_0 = np.random.randn(n_concept, n_dim)
    # z_0 = np.random.rand(n_concept, n_dim)
    noise = noise * np.random.randn(n_concept, n_dim)
    z_1 = z_0 + noise
    return z_0, z_1


def create_n_systems(n_epicentres=1, 
                epicentre_range=1,  
                n_dim=2, 
                num_concepts=200, 
                sigma=1, 
                noise_size=0.1, 
                n_systems = 1, 
                return_noisy = False, 
                plot = False, 
                rotation = True):
    """
    Function for creating multiple 'clumpy' Gaussian systems. 
    
    Arguments:
        n_epicentres: scalar indicating the number of gaussians from which 
        points are drawn per system
        epicentre_range: range within which gaussian means can be drawn
        (from +epicentre_range to -epicentre_range)
        n_dim: scalar indicating number of dimensions
        n_concepts: scalar indicating the number of concepts
        sigma: standard deviation of gaussians from which points are drawn
        noise_size: standard deviation of noise kernel used for transformation
        between systems
        n_systems: scalar indicating number of systems to be generated
        return_noisy: boolean indicating whether unrotated systems should be 
        returned
        plot: boolean, indicating whether to plot systems if n_dim = 2
    
    Returns:
        systems: list of length n, where list[i] is tensor of points in 
        system i
        return_noisy_X: if return_noisy=True, also returns a second list 
        of length n containing the unrotated versions of the systems, intended 
        for calculation of cieling mapping accuracy values. 
    """

    # Create X
    X_cov = np.zeros((n_dim, n_dim), float)
    np.fill_diagonal(X_cov, sigma)
    
    means = np.random.uniform(
        -epicentre_range, epicentre_range, size = (n_epicentres, n_dim)
        )

    X = []
    for i in range(num_concepts):
        mean = i % n_epicentres
        value = np.random.multivariate_normal(
            mean=means[mean], cov=X_cov,  size=1
            )
        X.append(value)
    X = np.array(X)
    X = np.squeeze(X)

    X_tensor = tf.expand_dims(
        tf.convert_to_tensor(X, dtype = tf.float32),0
        )

    systems = [X_tensor]
    return_noisy_X = [X_tensor]

    for i in range(n_systems-1):
        # Generate random rotation matrix
        random_rot_mat = sp.stats.special_ortho_group.rvs(n_dim)
    
        # Generate noisy X
        noisy_X = (X + np.random.multivariate_normal(
                mean = [0]*n_dim, cov = X_cov * noise_size,  size = num_concepts
                ))

        if return_noisy == True:
            noisy_X_tensor = tf.expand_dims(
                tf.convert_to_tensor(noisy_X, dtype = tf.float32),0
                )
            return_noisy_X.append(noisy_X_tensor)
    
        if rotation == True:
        # Create Y by rotating noisy X
            Y = np.matmul(random_rot_mat, noisy_X.T)
            Y = Y.T
        else:
            Y = noisy_X

        Y = tf.convert_to_tensor(Y,dtype = tf.float32)
        Y = tf.expand_dims(tf.convert_to_tensor(Y, dtype = tf.float32),0)
        systems.append(Y)

    #RP - only worked for 2 systems, adapted up to 3.
    if n_dim == 2 and plot == True and n_systems <= 3:
        fig = plt.figure()
        colors = ["#f64b4bff","#4b64f6ff",'g']
        labels = ["System A","System B","System C"]       
        for i in range(n_systems):
           plt.scatter(
            systems[i][:,:,0], systems[i][:,:,1], 
            color = colors[i], label = labels[i]
            ) 
        plt.title("Synthetic data")
        plt.legend()
        plt.show()

    if return_noisy == False:
        return systems
    
    elif return_noisy == True:
        return systems, return_noisy_X



def glove_word_and_glove_image(fp_intersect):
    """Load GloVe embeddings for words and images.

    The embeddings and vocabulary are returned in aligned order.

    Arguments:
        fp_intersect: A filepath to the intersect directory.

    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on images.
        vocab: A list of the intersection vocabulary.

    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_glove.840b-openimage.box.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = intersect_data['vocab_intersect']
    return z_0, z_1, vocab


def glove_word_and_glove_audio(fp_intersect):
    """Load GloVe embeddings for words and audio.

    The embeddings and vocabulary are returned in aligned order.

    Arguments:
        fp_intersect: A filepath to the intersect directory.

    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on audio.
        vocab: A list of the intersection vocabulary.

    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_glove.840b-audioset.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = intersect_data['vocab_intersect']
    return z_0, z_1, vocab


def glove_image_and_glove_audio(fp_intersect):
    """Load GloVe embeddings for images and audio.

    The embeddings and vocabulary are returned in aligned order.

    Arguments:
        fp_intersect: A filepath to the intersect directory.

    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on images.
        vocab_intersect: A list of the intersection vocabulary.

    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_openimage.box-audioset.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = intersect_data['vocab_intersect']
    return z_0, z_1, vocab
