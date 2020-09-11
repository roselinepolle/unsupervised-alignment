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

import os
import shutil
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import tensorflow as tf


DATA_WORD = 'glove.840b'
DATA_IMAGE = 'openimage.box'
DATA_AUDIO = 'audioset'
DATA_PIXEL = 'imagenet'

####=======================================================
#####=====================RP===============================
####=======================================================
def set_seed_np_tf(sd = None, upper_seed = 10000):
    if sd == None:
        sd_to_use = np.random.randint(0,upper_seed)
    else:
        sd_to_use = sd
    tf.random.set_seed(sd_to_use)
    np.random.seed(sd_to_use)
    return sd_to_use
    
def shuffle_system(X):
    #To get correct mapping map: do np.argsort(X_shuff_idx)
    n_concepts = X.shape[-2]
    X_shuff_idx = np.random.permutation(n_concepts)
    X_idx_map = np.argsort(X_shuff_idx)
    X_shuff = X[X_shuff_idx,:] 
    #=tf version
    #X_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
    #X_shuff = tf.expand_dims(tf.gather(tf.squeeze(X), X_shuff_idx), axis=0)
    return X_shuff,X_shuff_idx,X_idx_map
def get_batch(inputX, inputY, batch_size):
    duration = inputX.shape[-2]
    for i in range(0,duration//batch_size):
        idx = i*batch_size
        yield inputX[idx:idx+batch_size], inputY[idx:idx+batch_size]
def plot_systems_results(x,y,f_x,g_y,x_0,x_1):
    
    n_dim = x.shape[-1]
    
    
    if n_dim ==2:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
        #Before
        ax1.set_title('Before')
        ax1.scatter(x[:,0],x[:,1])
        ax1.scatter(y[:,0],y[:,1])
        #After x
        ax2.set_title('After x : gy')
        ax2.scatter(x[:,0],x[:,1])
        ax2.scatter(g_y[:,0],g_y[:,1])
        #After y
        ax3.set_title('After y : fx')
        ax3.scatter(y[:,0],y[:,1])
        ax3.scatter(f_x[:,0],f_x[:,1])
        #Noisy X
        ax4.set_title('Data')
        ax4.scatter(x_0[:,0],x_0[:,1])
        ax4.scatter(x_1[:,0],x_1[:,1])
        #Show
        plt.show()
    elif n_dim ==3:
        fig = plt.figure(figsize=(15., 5.))
        #Before
        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        ax1.set_title('Before')
        ax1.scatter(x[:,0],x[:,1],x[:,2])
        ax1.scatter(y[:,0],y[:,1],y[:,2])
        #After x
        ax2 = fig.add_subplot(1, 4, 2, projection='3d')
        ax2.set_title('After x : gy')
        ax2.scatter(x[:,0],x[:,1],x[:,2])
        ax2.scatter(g_y[:,0],g_y[:,1],g_y[:,2])
        #After y
        ax3 = fig.add_subplot(1, 4, 3, projection='3d')
        ax3.set_title('After y : fx')
        ax3.scatter(y[:,0],y[:,1],y[:,2])
        ax3.scatter(f_x[:,0],f_x[:,1],f_x[:,2])
        #Noisy X
        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        ax4.set_title('Data')
        ax4.scatter(x_0[:,0],x_0[:,1],x_0[:,2])
        ax4.scatter(x_1[:,0],x_1[:,1],x_1[:,2])
        #Show
        plt.show()
        

def flatten_list_of_list(list_of_list):
    flat_list = [item for sublist in list_of_list for item in sublist]
    return flat_list
####=======================================================
#####=====================FETCH RESULTS====================
####=======================================================
def get_acc_from_dict(dict_entry):
    acc_f1, acc_f5, acc_f10, acc_fhalf = dict_entry['acc_f1'],dict_entry['acc_f5'],dict_entry['acc_f10'],dict_entry['acc_fhalf']
    acc_g1, acc_g5, acc_g10, acc_ghalf = dict_entry['acc_g1'],dict_entry['acc_g5'],dict_entry['acc_g10'],dict_entry['acc_ghalf']
    return acc_f1, acc_f5, acc_f10, acc_fhalf,acc_g1, acc_g5, acc_g10, acc_ghalf
def get_losses_from_dict(dict_entry):
    loss_total, cycle_loss, dist_loss_f, dist_loss_g = dict_entry['loss_total'],dict_entry['cycle_loss'],dict_entry['dist_loss_f'],dict_entry['dist_loss_g']
    return loss_total, cycle_loss, dist_loss_f, dist_loss_g    

def plot_df_columns(axes,df,column_names):
    #Plot the columns on axes
    for i in range(len(axes)):
        axes[i].set_title(column_names[i])
        axes[i].plot(list(df[column_names[i]]))

def get_df(df,exp=None,run=None,restart=None,epoch=None,new_f = None,new_g = None):
    #Filter the input df to get a sub-df
    df2 = df
    if not exp == None:
        df2 = df2[(df2['experiment_name']==exp)]
    if not run == None:
        df2 = df2[(df2['run']==run)]
    if not restart == None:
        df2 = df2[(df2['restart']==restart)]
    if not epoch == None:
        df2 = df2[(df2['epoch']==epoch)]
    if not new_f == None:
        df2 = df2[(df2['new_f']==new_f)]
    if not new_g == None:
        df2 = df2[(df2['new_g']==new_g)]
    return df2
def sns_cat_data_to_df(categories,data_list,cat_name='cat',val_name = 'values'):
    #Put data in df for seaborn use
    dict_list = []
    for c in range(len(categories)):
        data = data_list[c]
        for i in range(len(data)):
            dict_list.append({cat_name:categories[c],val_name:data[i]})
    df = pd.DataFrame(dict_list)
    return df      
    
    

####=======================================================
#####=====================PATHS/SAVE=======================
####=======================================================
def save_models(model_f,model_g,fp):
    create_folder(fp+ '/model_f')
    create_folder(fp+ '/model_g')
    model_f.save_weights(fp+ '/model_f/model_f')
    model_g.save_weights(fp+ '/model_g/model_g')
    
def create_save_folder(fp_save,override,
                       n_systems,n_concepts,noise,emb_dim,n_epicentres,linearsep,max_restart,max_epoch,batch_size,
                       name = None):
    if name == None:
        template_save = 'DATA_nsys{0}ncon{1}noise{2:.3f}emb_dim{3}nepi{4}linearsep{5}_NN_mrest{6}mepo{7}bsz{8}'
        save_folder = template_save.format(n_systems,n_concepts,noise,emb_dim,n_epicentres,linearsep,max_restart,max_epoch,batch_size)
    else:
        save_folder = name
    fp_save_runs = str(fp_save / Path(save_folder))
    i = 0
    if create_folder(fp_save_runs,override)==False and saved_folder_virgin(fp_save_runs)==False:
        while create_folder(fp_save_runs+ '_' + str(i))==False and saved_folder_virgin(fp_save_runs+ '_' + str(i))==False:
            i +=1
        fp_save_runs = fp_save_runs + '_' + str(i)
    fp_save_models = fp_save_runs +'/models'
    create_folder(fp_save_runs +'/models')
    return fp_save_runs,fp_save_models
def create_folder(folder_path,override = False):
    created = False
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        created = True
    elif override:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        created = True
    return created
def saved_folder_virgin(folder_path):
    #only empty 'models' dir allowed
    is_virgin = False
    if folder_is_empty(folder_path):
        is_virgin = True
    elif [f for f in os.listdir(folder_path) if not f.startswith('.')]==['models','dict_params.pickle']:
        if folder_is_empty(folder_path/Path('models')):
            is_virgin = True
    return is_virgin
def folder_is_empty(folder_path):
    if [f for f in os.listdir(folder_path) if not f.startswith('.')] == []:
        is_empty = True
    else: 
        is_empty = False
    return is_empty  



####========================================================
#####=====================BEFORE============================
####========================================================

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


def calculate_ceiling(z_1p, z_1):

    """Calculate proportion of correct mappings"""
    z_1p = tf.squeeze(z_1p)
    z_1 = tf.squeeze(z_1)
    n_concept = z_1.shape[0]
    correct = range(n_concept)

    map_idx = assign_neighbor(z_1p, z_1)
    compare = np.sum(map_idx == correct)
    
    return compare/n_concept


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


