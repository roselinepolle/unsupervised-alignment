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

"""Use synthetic data to evaluate alignment algorithm."""

import copy
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
# import tensorflow_graphics as tfg
tfd = tfp.distributions
kl = tfd.kullback_leibler

import utils




def main(fp_repo):
    """Run script."""
    # Settings
    tf.keras.backend.set_floatx('float64')
    fp_intersect = fp_repo / Path('python', 'assets', 'intersect')

    # Load real embeddings.
    z_0, z_1, vocab = embeddings.glove_word_and_glove_image(fp_intersect)
    # z_0, z_1, vocab = embeddings.glove_word_and_glove_audio(fp_intersect)
    # z_0, z_1, vocab = embeddings.glove_image_and_glove_audio(fp_intersect)

    # Pre-process embeddings.
    z_0 = utils.preprocess_embedding(z_0)
    z_1 = utils.preprocess_embedding(z_1)

    # Shuffle.
    # z_1_shuffle = z_1
    # y_idx_map = np.arange(z_0.shape[0])
    n_concept = z_0.shape[0]
    idx_rand = np.random.permutation(n_concept)
    z_1_shuffle = z_1[idx_rand, :]
    # # Determine mapping key.
    y_idx_map = np.argsort(idx_rand)
    # Verify mapping key.
    np.testing.assert_array_equal(z_1, z_1_shuffle[y_idx_map, :])

    # idx_known = np.array([151, 82, 197])
    # idx_known = np.arange(0, 100, 10)
    learn_mapping_v2(z_0, z_1_shuffle, y_idx_map)