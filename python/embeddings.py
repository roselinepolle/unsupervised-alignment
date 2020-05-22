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

    if n_dim == 2 and plot == True:
        fig = plt.figure()

        handles = []
        plt.scatter(
            systems[0][:,:,0], systems[0][:,:,1], 
            color = "#f64b4bff", label = "System A"
            )
        plt.scatter(
            systems[1][:,:,0], systems[1][:,:,1], 
            color = "#4b64f6ff", label = "System B"
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
