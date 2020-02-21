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

"""A module of alignment algorithms.

Each algorithm learns a mapping from system X to system Y.

"""

import copy

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Add
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

import utils

def bdr_0_0_1(x, y, y_idx_map=None):
    """Align using an algorithm loosely inspired by CycleGan.

    Arguments:
        x: A 2D NumPy array indicating the items in system X.
            shape=(n_concept, n_dim_x)
        y: A 2D NumPy array indicating the items in system Y.
            shape=(n_concept, n_dim_y)
        y_idx_map (optional): Array indicating how y should be permuted
            in order to produce the correct alignment. This is used to
            compute mapping accuracy.

    Outputs:
        TODO

    """
    # Settings.
    n_restart = 1000
    max_epoch = 30
    batch_size = np.minimum(100, x.shape[0])
    max_patience = 5
    loss_distr_scale = tf.constant(1., dtype=K.floatx())
    loss_cycle_scale = tf.constant(10000.0, dtype=K.floatx())
    gmm_scale = tf.constant(.01, dtype=K.floatx())

    # Setup.
    n_concept = x.shape[0]
    n_batch = int(np.ceil(n_concept / batch_size))
    n_pair = int(np.floor(n_concept / 2))
    n_dim_x = x.shape[1]
    n_dim_y = y.shape[1]
    n_sample = n_concept * 10  # TODO

    optimizer = tf.keras.optimizers.Adam()
    mse_loss_func = tf.keras.losses.MeanSquaredError()

    # Pre-assemble GMM.
    seed = 124325

    # @tf.function
    def train_batch_step(x_batch, y_batch, x_all, y_all, curr_idx):
        with tf.GradientTape(persistent=True) as tape_cycle:
            x_c = model_g(model_f(x_batch))
            y_c = model_f(model_g(y_batch))
            cycle_loss = cycle_loss_func(x_batch, y_batch, x_c, y_c, loss_cycle_scale)
        gradients_cycle_f = tape_cycle.gradient(
            cycle_loss, model_f.trainable_variables
        )
        gradients_cycle_g = tape_cycle.gradient(
            cycle_loss, model_g.trainable_variables
        )
        del tape_cycle

        optimizer.apply_gradients(
            zip(gradients_cycle_f, model_f.trainable_variables)
        )
        optimizer.apply_gradients(
            zip(gradients_cycle_g, model_g.trainable_variables)
        )

        return cycle_loss

    # @tf.function
    def train_full_step(x_all, y_all, gmm_x_samples, gmm_x_log_prob, gmm_y_samples, gmm_y_log_prob, loss_distr_scale):
        with tf.GradientTape() as tape_f0:
            f_x = model_f(x_all)
            dist_loss_f = loss_distr_scale * distribution_loss(f_x, gmm_y_samples, gmm_y_log_prob, gmm_scale)
        gradients_target_fx = tape_f0.gradient(dist_loss_f, model_f.trainable_variables)

        with tf.GradientTape() as tape_g0:
            g_y = model_g(y_all)
            dist_loss_g = loss_distr_scale * distribution_loss(g_y, gmm_x_samples, gmm_x_log_prob, gmm_scale)
        gradients_target_gy = tape_g0.gradient(dist_loss_g, model_g.trainable_variables)

        # Combine gradients.
        optimizer.apply_gradients(
            zip(gradients_target_fx, model_f.trainable_variables)
        )
        optimizer.apply_gradients(
            zip(gradients_target_gy, model_g.trainable_variables)
        )

        return dist_loss_f, dist_loss_g

    loss_best = np.inf
    best_model_f = MLP(n_dim_x, n_dim_y)
    best_model_g = MLP(n_dim_y, n_dim_x)
    dist_loss_g_best = np.inf
    dist_loss_f_best = np.inf
    for i_restart in range(n_restart):
        print('Restart {0}'.format(i_restart))

        # Use last few restarts to fine-tune best model.
        thresh = .7  # .7 TODO
        if i_restart < (n_restart - 10):  # TODO
            rand_val = np.random.rand(2)
            if rand_val[0] < thresh:
                model_f = MLP(n_dim_x, n_dim_y)
                print('    New model_f')
            else:
                model_f = best_model_f
            if rand_val[1] < thresh:
                model_g = MLP(n_dim_y, n_dim_x)
                print('    New model_g')
            else:
                model_g = best_model_g
        else:
            model_f = best_model_f
            model_g = best_model_g

        f_x = model_f(x).numpy()
        g_y = model_g(y).numpy()
        display_state(x, y[y_idx_map], f_x, g_y[y_idx_map])
        plt.show(block=False)
        plt.pause(.1)
        plt.close()

        # Start algorithm.
        template_loss = '    Epoch {0} Loss | total: {1:.5g} | cycle: {2:.3g} | f_dist: {3:.3g} | g_dist: {4:.3g}'
        template_acc = '        {0} Accuracy | 1: {1:.2f} | 5: {2:.2f} | 10: {3:.2f} | half: {4:.2f}'

        # Initialize.
        gmm_x = assemble_gmm(x, gmm_scale)
        gmm_y = assemble_gmm(y, gmm_scale)
        gmm_x_samples = tf.constant(x, dtype=K.floatx())
        gmm_y_samples = tf.constant(y, dtype=K.floatx())
        gmm_x_log_prob = tf.reduce_mean(gmm_x.log_prob(gmm_x_samples), axis=0)
        gmm_y_log_prob = tf.reduce_mean(gmm_y.log_prob(gmm_y_samples), axis=0)

        for i_epoch in range(max_epoch):
            idx_batch = np.random.permutation(n_concept)
            idx_start = 0
            for i_batch in range(n_batch):
                idx_end = idx_start + batch_size
                curr_idx = idx_batch[idx_start:idx_end]
                # Perform gradient update.
                cycle_loss = train_batch_step(
                    x[curr_idx], y[curr_idx], x, y, tf.constant(curr_idx)
                )
                idx_start = idx_end
            dist_loss_f, dist_loss_g = train_full_step(x, y, gmm_x_samples, gmm_x_log_prob, gmm_y_samples, gmm_y_log_prob, loss_distr_scale)
            loss_total = dist_loss_f + dist_loss_g + cycle_loss

            # Project concept using best model x -> y.
            f_x = model_f(x).numpy()
            g_y = model_g(y).numpy()

            if y_idx_map is not None:
                acc_f1, acc_f5, acc_f10, acc_fhalf = utils.mapping_accuracy(f_x, y[y_idx_map])
                acc_g1, acc_g5, acc_g10, acc_ghalf = utils.mapping_accuracy(g_y[y_idx_map], x)

            # Evaluate performance.
            if np.mod(i_epoch, 2) == 0:
                display_state(x, y[y_idx_map], f_x, g_y[y_idx_map])
                plt.show(block=False)
                plt.pause(.01)
                plt.close()

                print(
                    template_loss.format(
                        i_epoch+1, loss_total, cycle_loss,
                        dist_loss_f, dist_loss_g
                    )
                )
                if y_idx_map is not None:
                    print(
                        template_acc.format(
                            'f(x)', acc_f1, acc_f5, acc_f10, acc_fhalf
                        )
                    )
                    print(
                        template_acc.format(
                            'g(y)', acc_g1, acc_g5, acc_g10, acc_ghalf
                        )
                    )

        print(
            template_loss.format(
                i_epoch+1, loss_total, cycle_loss,
                dist_loss_f, dist_loss_g
            )
        )
        if dist_loss_f < dist_loss_f_best:
            dist_loss_f_best = dist_loss_f
            best_model_f = model_f
            print('    Beat best model_f.')

        if dist_loss_g < dist_loss_g_best:
            dist_loss_g_best = dist_loss_g
            best_model_g = model_g
            print('    Beat best model_g.')

        # Project concept using best model x -> y.
        f_x = best_model_f(x).numpy()
        g_y = best_model_g(y).numpy()

        if y_idx_map is not None:
            acc_f1, acc_f5, acc_f10, acc_fhalf = utils.mapping_accuracy(f_x, y[y_idx_map])
            acc_g1, acc_g5, acc_g10, acc_ghalf = utils.mapping_accuracy(g_y[y_idx_map], x)
            print(
                template_acc.format('f(x)', acc_f1, acc_f5, acc_f10, acc_fhalf)
            )
            print(
                template_acc.format('g(y)', acc_g1, acc_g5, acc_g10, acc_ghalf)
            )
            if acc_f1 > .95 and acc_g1 > .95:
                break

    f_x = best_model_f(x).numpy()
    g_y = best_model_g(y).numpy()

    acc_f1, acc_f5, acc_f10, acc_fhalf = utils.mapping_accuracy(f_x, y[y_idx_map])
    acc_g1, acc_g5, acc_g10, acc_ghalf = utils.mapping_accuracy(g_y[y_idx_map], x)
    print(
        template_acc.format('f(x)', acc_f1, acc_f5, acc_f10, acc_fhalf)
    )
    print(
        template_acc.format('g(y)', acc_g1, acc_g5, acc_g10, acc_ghalf)
    )

    display_state(x, y[y_idx_map], f_x, g_y[y_idx_map])
    plt.show()


class MLP(Model):
    """Model that aligns two domains."""

    def __init__(self, n_dim_0, n_dim_1):
        """Initialize."""
        n_hidden = 100
        # NOTE: elu sometimes works
        super(MLP, self).__init__()
        # self.d1 = Dense(n_dim_0, activation='relu', kernel_initializer='glorot_uniform')
        # self.d2 = Dense(n_hidden, activation='relu', kernel_initializer='glorot_uniform')
        self.d1 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        # self.drop1 = Dropout(0.2)
        self.d2 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        # self.drop2 = Dropout(0.2)
        self.d3 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        # self.d4 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        # self.d5 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        # self.d6 = Dense(n_hidden, activation='tanh', kernel_initializer='glorot_uniform')
        self.d5 = Dense(n_dim_1, activation='tanh')
        # self.angles = tf.Variable(
        #     initial_value=tf.zeros(n_dim_1-1, dtype=K.floatx()),
        #     trainable=True,
        #     dtype=K.floatx(),
        #     name='angles'
        # )  # TODO generalize

    def call(self, x):
        """Call."""
        x1 = self.d1(x)
        # x = self.drop1(x)
        x2 = self.d2(x1)
        # x = self.drop2(x)
        x3 = self.d3(x2)  # TODO was off, turned on for 5D
        # x = self.d4(x)
        # x = self.d5(x)
        # x = self.d6(x)
        # added = Add()([x1, x2])
        x4 = self.d5(x3)
        # r = tfg.geometry.transformation.rotation_matrix_2d.from_euler(
        #     self.angles
        # )  # TODO generalize
        # x5 = tf.linalg.matmul(x4, r)
        return x4


# @tf.function
def cycle_loss_func(x, y, x_c, y_c, loss_cycle_scale):
    """Cycle loss.

    Compute the L1 distance between original and return point for each
    model.

    """
    # TODO L1 or L2 or L2**2?
    # d_x = tf.abs(x - x_c)
    d_x = (x - x_c)**2
    d_x = tf.reduce_sum(d_x, axis=1)

    # d_y = tf.abs(y - y_c)
    d_y = (y - y_c)**2
    d_y = tf.reduce_sum(d_y, axis=1)

    loss = loss_cycle_scale * tf.constant(.5, dtype=K.floatx()) * (tf.reduce_mean(d_x) + tf.reduce_mean(d_y))
    return loss


def display_state(x, y, fx, gy):
    """Display current mapping state."""
    # Settings.

    n_dim = x.shape[1]

    if n_dim == 2:
        fig, ax = plt.subplots(figsize=(7., 7.))
        ax = plt.subplot(2, 2, 1)
        ax.scatter(x[:, 0], x[:, 1], c='b', alpha=.25)
        label_points(ax, x)
        ax.set_title('System X')
        ax.set_aspect('equal')

        ax = plt.subplot(2, 2, 3)
        ax.scatter(gy[:, 0], gy[:, 1], c='r', alpha=.25)
        label_points(ax, gy)
        ax.set_title('System g(Y)')
        ax.set_aspect('equal')

        ax = plt.subplot(2, 2, 2)
        ax.scatter(y[:, 0], y[:, 1], c='r', alpha=.25)
        label_points(ax, y)
        ax.set_title('System Y')
        ax.set_aspect('equal')

        ax = plt.subplot(2, 2, 4)
        ax.scatter(fx[:, 0], fx[:, 1], c='b', alpha=.25)
        label_points(ax, fx)
        ax.set_title('System f(X)')
        ax.set_aspect('equal')

    elif n_dim == 3:
        fig = plt.figure(figsize=(10., 10.))
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='b')
        label_points(ax, x)
        ax.set_title('System X')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(gy[:, 0], gy[:, 1], gy[:, 2], c='r')
        label_points(ax, gy)
        ax.set_title('System g(Y)')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], c='r')
        label_points(ax, y)
        ax.set_title('System Y')

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(fx[:, 0], fx[:, 1], fx[:, 2], c='b')
        label_points(ax, fx)
        ax.set_title('System f(X)')


def label_points(ax, x, offset_y=.02):
    """Label points."""
    n_point = np.minimum(x.shape[0], 50)
    n_dim = x.shape[1]

    if n_dim == 2:
        for i_point in range(n_point):
            ax.text(
                x[i_point, 0], x[i_point, 1] + offset_y, '{0}'.format(i_point),
                horizontalalignment='center', verticalalignment='center'
            )
    elif n_dim == 3:
        for i_point in range(n_point):
            ax.text(
                x[i_point, 0], x[i_point, 1], x[i_point, 2], '{0}'.format(i_point),
                horizontalalignment='center', verticalalignment='center'
            )


def assemble_gmm(x, gmm_scale):
    """Assemble GMM."""
    n_concept = tf.constant(x.shape[0], dtype=K.floatx())
    tf_w = tf.ones([n_concept], dtype=K.floatx()) / n_concept

    # Weight by distance to neighbors.  # TODO
    # k = 5  # TODO
    # neigh = NearestNeighbors(n_neighbors=k + 1)
    # neigh.fit(x)
    # # Determine which concepts of y are closest for each point in f_x.
    # d, indices = neigh.kneighbors(x)
    # d = np.sum(d, axis=1)
    # w = d / np.sum(d)
    # tf_w = tf.constant(w, dtype=K.floatx())
    

    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=tf_w),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.constant(x, dtype=K.floatx()),
            scale_identity_multiplier=gmm_scale * tf.ones([n_concept], dtype=K.floatx())
        )
    )
    return gmm


def distribution_loss(fx, gmm_y_samples, gmm_y_log_prob, gmm_scale):
    """Distribution loss.

    Use upperbound of GMM KL divergence approximation.

    Assumes that incoming `fx` and `y` is all of the data.

    """
    n_concept = tf.constant(fx.shape[0], dtype=K.floatx())
    w = tf.ones([n_concept], dtype=K.floatx()) / n_concept

    # Weight by neighbor distance. TODO
    # r = tf.reduce_sum(fx*fx, 1)
    # # Turn r into column vector.
    # r = tf.reshape(r, [-1, 1])
    # d_mat = r - 2*tf.matmul(fx, tf.transpose(fx)) + tf.transpose(r)
    # k = 5  # TODO
    # d_mat = tf.sort(d_mat, axis=1)
    # d_mat = d_mat[:, 0:k]
    # w = tf.reduce_sum(d_mat, axis=1)
    # w = w / tf.reduce_sum(d_mat)

    gmm_fx = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=w),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=fx,
            scale_identity_multiplier=gmm_scale * tf.ones([n_concept], dtype=K.floatx())
        )
    )

    loss_kl = gmm_y_log_prob - tf.reduce_mean(
        gmm_fx.log_prob(gmm_y_samples), axis=0
    )
    return loss_kl
