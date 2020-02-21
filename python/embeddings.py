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
