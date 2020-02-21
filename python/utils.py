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

"""Module of common utilities."""

from pathlib import Path
import os

import numpy as np
import pickle
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


DATA_WORD = 'glove.840b'
DATA_IMAGE = 'openimage.box'
DATA_AUDIO = 'audioset'
DATA_PIXEL = 'imagenet'


def load_synthetic_embeddings_1_0(noise=.1, n_dim=2, n_concept=200, seed=7849):
    """Load synthetic embeddings."""
    # Create synthetic embeddings.
    np.random.seed(seed)
    z_0 = np.random.randn(n_concept, n_dim)
    # z_0 = np.random.rand(n_concept, n_dim)
    noise = noise * np.random.randn(n_concept, n_dim)
    z_1 = z_0 + noise
    return z_0, z_1


def load_synthetic_embeddings_2_0(noise=.1, n_dim=2, n_concept=200, seed=7849):
    """Load synthetic embeddings."""
    # Create synthetic embeddings.
    np.random.seed(seed)
    z_0 = np.random.randn(n_concept, n_dim)
    z_0[:, 0] = z_0[:, 0] / 2
    # z_0 = np.random.rand(n_concept, n_dim)
    noise = noise * np.random.randn(n_concept, n_dim)
    z_1 = z_0 + noise
    return z_0, z_1


def load_real_embeddings(fp_repo):
    """Load real embeddings."""
    fp_intersect_word_image = build_intersect_path(
        [DATA_IMAGE, DATA_AUDIO], fp_repo, include_aoa=False
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    return z_0, z_1


def load_real_based_embeddings(fp_repo, seed=7849):
    """Real-based embedding."""
    n_dim = 2
    np.random.seed(seed)
    noise = .01
    fp_intersect_word_image = build_intersect_path(
        [DATA_WORD, DATA_IMAGE], fp_repo, include_aoa=False
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_1']
    # z_0 = LocallyLinearEmbedding(n_neighbors=50, n_components=3).fit_transform(z_0)
    z_0 = TSNE(n_components=n_dim).fit_transform(z_0)
    z_0 = preprocess_embedding(z_0)
    z_1 = z_0 + noise * np.random.randn(z_0.shape[0], z_0.shape[1])
    return z_0, z_1


def preprocess_embedding(z):
    """Pre-process embedding."""
    # Normalize coordinate system.
    # gmm = GaussianMixture(n_components=1, covariance_type='full')
    # gmm.fit(z)
    # mu = gmm.means_[0]
    # sigma = gmm.covariances_[0]
    gmm = GaussianMixture(n_components=1, covariance_type='spherical')
    gmm.fit(z)
    mu = gmm.means_[0]
    sigma = gmm.covariances_[0]
    # z_norm = (z - mu) / sigma
    z_norm = z - mu
    max_val = np.max(np.abs(z_norm))
    z_norm = z_norm / max_val
    z_norm = z_norm / 3  # TODO experimenting with different scales here
    z_norm + 1  # TODO
    return z_norm


def build_intersect_path(dataset_list, fp_base, include_aoa=False):
    """Build filepath."""
    fp_intersect = fp_base / Path('intersect')
    fn = 'intersect_'
    for i_dataset in dataset_list:
        fn = fn + i_dataset + '-'
    # Remove last dash.
    fn = fn[0:-1]
    if include_aoa:
        fn = fn + '-aoa'
    fn = fn + '.p'
    fp_intersect = fp_intersect / Path(fn)
    return fp_intersect


def assign_neighbor(z_1p, z_1):
    """Assign mapping."""
    n_concept = z_1.shape[0]
    map_idx = np.zeros([n_concept], dtype=int)

    for i_concept in range(n_concept):
        z_1p_i = np.expand_dims(z_1p[i_concept], axis=0)
        # Determine closest point in Euclidean space.
        d = np.sum((z_1p_i - z_1)**2, axis=1)**.5
        map_idx[i_concept] = np.argsort(d)[0]

    return map_idx


def mapping_accuracy(f_x, y):
    """Compute mapping accuracy.

    Assumes inputs f_x and y are aligned.

    """
    n_concept = f_x.shape[0]
    n_half = int(np.ceil(n_concept / 2))

    # Create nearest neighbor graph for y.
    neigh = NearestNeighbors(n_neighbors=n_half)
    neigh.fit(y)
    # Determine which concepts of y are closest for each point in f_x.
    _, indices = neigh.kneighbors(f_x)

    dmy_idx = np.arange(n_concept)
    dmy_idx = np.expand_dims(dmy_idx, axis=1)

    locs = np.equal(indices, dmy_idx)

    is_correct_half = np.sum(locs[:, 0:n_half], axis=1)
    is_correct_10 = np.sum(locs[:, 0:10], axis=1)
    is_correct_5 = np.sum(locs[:, 0:5], axis=1)
    is_correct_1 = locs[:, 0]

    acc_half = np.mean(is_correct_half)
    acc_10 = np.mean(is_correct_10)
    acc_5 = np.mean(is_correct_5)
    acc_1 = np.mean(is_correct_1)
    return acc_1, acc_5, acc_10, acc_half


def load_hierarchical_embeddings():
    """Load."""
    # Settings.
    noise = .1

    k = 3
    n_dim = 2
    np.random.seed(7877)

    c_0 = 4 * np.random.randn(k, n_dim)
    c_1 = 2 * np.random.randn(k**2, n_dim)
    z_0 = .5 * np.random.randn(k**3, n_dim)

    idx_start = 0
    counter = 0
    for ik in range(k):
        idx_end = idx_start + k**2
        z_0[idx_start:idx_end] = c_0[ik, :] + z_0[idx_start:idx_end]

        idx_start_j = idx_start
        for jk in range(k):
            idx_end_j = idx_start_j + k
            z_0[idx_start_j:idx_end] = c_1[counter, :] + z_0[idx_start_j:idx_end]
            counter = counter + 1
            idx_start_j = idx_end_j

        idx_start = idx_end

    z_1 = z_0 + (noise * np.random.randn(k**3, n_dim))
    return z_0, z_1


