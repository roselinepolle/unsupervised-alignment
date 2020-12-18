#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:59:09 2020

@author: roseline

This code is related to the calculation of local features, and finding correspondences. 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import itertools
import time
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.spatial import distance
#Display
from IPython.display import display, clear_output
#Custom
import embeddings as emb


#====COMMON TO ALL=====
def get_corres_data(X,Y,DXY,Ncorr):
    "Finds the idx of Ncorr closest points (smallest values) in DXY (N,N)."
    "Does not exclude value 0"
    #==Gets smallest indices
    matches = smallestN_indices(DXY, Ncorr)
    idx_X = list(matches[:,0]) #Idx in space X of these pairs
    idx_Y = list(matches[:,1])
    #==Subset data tf np
    X_pretrain = filter_rows(X,idx_X)
    Y_pretrain = filter_rows(Y,idx_Y)
    return X_pretrain,Y_pretrain, matches

def check_corresp(matches,y_idx_map,bPrint = True):
    idx_X = matches[:,0]
    idx_Y = matches[:,1]
    #==== concat real corres and 10 closest indices
    corr = y_idx_map[:,None] 
    corr = corr[idx_X] 
    mat = np.concatenate((idx_X[:,None],idx_Y[:,None],corr),axis = 1)
    n_right = len(np.where(mat[:,1]==mat[:,2])[0])
    n_tot = mat.shape[0]
    acc = n_right/n_tot
    if bPrint:
        print("%d right over %d tot" % (n_right,n_tot))
    return acc , mat

#***********************************************
#============PRETRAINING ANGLE K/RADIUS===========
#*********************************************** 
def pts_no_knn(feat_A,pad=-1):
    idx_pad = np.where((feat_A[:,0]==pad))[0]
    n_pad = len(idx_pad)
    return idx_pad,n_pad
def replace_with_pads_before_D(feat_A,feat_B,val_A,val_B):
    "Makes a deep copy"
    pad_A = np.array([-1,] * 8) 
    pad_B = np.array([-1,] * 8)
    pad_A[-1] = pad_A[-1] + val_A
    pad_B[-1] = pad_B[-1] + val_B
    idx_pad_A,n_pad_A = pts_no_knn(feat_A,-1)
    idx_pad_B,n_pad_B = pts_no_knn(feat_B,-1)
    feat_A_pad = feat_A.copy()
    feat_B_pad = feat_B.copy()
    feat_A_pad[idx_pad_A] = pad_A
    feat_B_pad[idx_pad_B] = pad_B
    return feat_A_pad,feat_B_pad
def get_DXY_angles(X,Y,k=None,radius = None,bpad = True,bplot = False,bPrint = True):
    #Get Distance matrix
    feat_X = get_X_feat_angles(X,k=k,radius = radius,bPrint =bPrint)
    feat_Y = get_X_feat_angles(Y,k=k,radius = radius,bPrint = bPrint)
    dist_feat_XY = distance.cdist(feat_X,feat_Y)  #np.linalg.norm(feat_X[:, None] - feat_Y[None, :],axis=(2))
    if radius is not None and bpad:
        #Padded version
        val_A = 10e6
        val_B = 10e12
        feat_X_pad,feat_Y_pad = replace_with_pads_before_D(feat_X,feat_Y,val_A,val_B)
        dist_feat_XY = distance.cdist(feat_X_pad,feat_Y_pad)
    if bplot:
        plt.figure()
        plt.imshow(dist_feat_XY)
        plt.colorbar()
        plt.show()
    return dist_feat_XY
def get_X_feat_angles(A,k=None,radius = None,bPrint = True):
    N = A.shape[0] 
    #0 - Check either k or radius is specified
    if all(v is None for v in {k, radius}):
        raise ValueError('Specify eaither a fixed k or the radius')
    #1 - Get Ris (neighbours of each i). (N, maxk or k)
    Ri , Ri_idx = get_Ris(A,k) if k is not None else get_Ris_radius(A,radius)
    #2 - Get Features for all posible pairs.
    normals = get_normals(Ri , Ri_idx,k = k,radius = radius) 
    f1s,f2s,f3s = get_features(A,normals) #f1s[t,s] catch f1 for pair (t,s)
    all_pairs_feat,all_pairs_idx = get_feat_all_pairs((f1s,f2s,f3s)) #(N,N,3).fetch (t,s) to get feature of pair (f1,f2,f3)
    #3 - Get pairs for each i
    pairs = get_pairs(Ri_idx,k = k,radius = radius) 
    #4 - Loop through each point. fetch features and construct feat_vect
    X_feat_vec = np.zeros((N,8))
    #start = time.time()
    for i in range(N):
        #Get all features for i (shape Npairsi x 3)
        feat_i = get_feat_i(i,pairs,all_pairs_idx,all_pairs_feat)
        #Get threshold
        thresh_i = np.mean(feat_i,axis = 0)
        #Get feat vector from this
        feat_vec_i = get_feat_vec_i(feat_i,thresh_i) #returns vec of -1 if no neighbours
        X_feat_vec[i,:] = feat_vec_i
    #Check how many null values (no neighbours)
    if radius is not None and bPrint:
        idx_pad,n_pad = pts_no_knn(X_feat_vec)
        if n_pad > 0.1*N: #more than 10%
            print("WARNING : Many points don't have neighbours, increase radius.")
    return X_feat_vec
def get_feat_all_pairs(features_A):
    f1s,f2s,f3s = features_A
    N = f1s.shape[0]
    all_pairs_idx = list(itertools.combinations(range(N), 2)) #all possible pairs
    all_pairs_feat = []
    for (t,s) in all_pairs_idx:
        all_pairs_feat.append([f1s[t,s],f2s[t,s],f3s[t,s]])
    return np.array(all_pairs_feat),np.array(all_pairs_idx)
def get_V(A):
    #As described in paper
    V = A[:, None] - A[None, :]
    V_norm= np.linalg.norm(V, axis=(2))
    np.fill_diagonal(V_norm, 1) #To avoid error when dividing
    V = V / V_norm[:,:,None]
    return V, V_norm
def get_feat_vec_i(feat_i,thresh_i):
    "Return the vector (feat dim,) corresponding to the local features of point i"
    "feat_i : (n_pairs_i,3) : Features values for each pair in Ri (knn of i)"
    "tresh_i : (3,) : Thresholds value. if feat above => =1, else 0"
    #Fixed bit code
    bit_code = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,1,1],[1,0,1],[1,0,0]])
    feat_dim = bit_code.shape[0]
    # Return ve of -1 if len(0)    
    if np.array_equal(feat_i,np.array([-1,-1,-1])):
        feat_vec_i =np.full((feat_dim,),-1) ##If radius too small for neighbours, just return a vect of -1
    else:
        code = (feat_i > thresh_i).astype(int)
        #Find index of rows of code in bit code
        X,searched_values = bit_code, code
        one_hot_idxes =  get_rows_idx(X,searched_values)
        #transform in one hots. n_pairs_i,feat_dim). one one-hot vec per row representing each pair
        one_hot = make_one_hot(one_hot_idxes,feat_dim)
        #Sum all rows to get final vect
        feat_vec_i = np.sum(one_hot,axis=0)
    return feat_vec_i

def get_features(A,normals):
    #normals (N,2). Normals of each Ri plane.
    V, V_norm = get_V(A)
    ntns = (normals[None,:,:] @  normals[:,:,None]).squeeze()
    #Get f1
    f1s = np.maximum(ntns,-ntns)
    #Get f2
    ntv = (V @ normals[:,:,None])
    ntv = tf.squeeze(ntv) if tf.is_tensor(ntv) else ntv.squeeze()
    ones = np.full((ntv.shape),1,dtype = np.float32)
    max_ntv = np.minimum(np.maximum(ntv,-ntv),ones) #some 1.000001 vaues
    f2s = np.abs(np.arccos(max_ntv)-np.arccos(max_ntv.T))
    #Get f3
    f3s = V_norm
    return f1s,f2s,f3s 

def get_feat_i(i,pairs,all_pairs_idx,all_pairs_feat):
    "Returns a (n_pairs_i,3) vector of features for pairs in knn of i"
    pairs_i = np.array(pairs[i])
    if len(pairs_i)==0: #if no neighbours
        feat_i = np.array([-1,-1,-1])
    else:
        pairs_idx = get_rows_idx(all_pairs_idx,pairs_i)
        feat_i = all_pairs_feat[pairs_idx]
    return feat_i

def get_normals(Ris,Ris_idx,k = None,radius = None):
    "Ris: (N , maxk or k , 2).pts coordinates of neighbours." 
    if all(v is None for v in {k, radius}):
        raise ValueError('Specify eaither a fixed k or the radius')
    if k is not None: #Fixed k algo
        mean_vecs = np.sum(Ris,1) / k #one for each Ri as we want to center those around 0
        X = Ris - mean_vecs[:,None,:] #remove centroids
        X = tf.transpose(X,perm=[0,2,1])
        u, s, vh = np.linalg.svd(X)
        normals = u[:,:,1] #/np.linalg.norm(u[:,:,1])
    elif radius is not None:
        freqs = get_freqs(Ris_idx)
        ks = freqs[:,1][:,None] #np.min(np.count_nonzero(Ris,1),1)[:,None]
        mean_vecs = np.sum(Ris,1) / ks
        X = Ris - mean_vecs[:,None,:] #remove centroids
        maxk = np.max(freqs[:,1])
        X = replace_pad_zeros(X,-mean_vecs,maxk) #replace pad values with 0 again
        #Get normal bulk code
        XT = tf.transpose(X,perm=[0,2,1])
        u, s, vh = np.linalg.svd(XT)
        normals = u[:,:,1]
    return normals #size (nconc,2)
def get_pairs(Ri_idx,k = None,radius = None):
    #TODO - include i in pairs ?
    if all(v is None for v in {k, radius}):
        raise ValueError('Specify eaither a fixed k or the radius')
    freqs = get_freqs(Ri_idx)
    pairs = []
    for i,ri_idx in enumerate(Ri_idx): 
        if radius is not None:
            k = freqs[i][1]
        list_idx = np.sort(ri_idx[1:k]) #TODO - this does not inclde k, sort so always smallest indice first in pair
        l = list(itertools.combinations(list_idx, 2)) 
        pairs.append(l)
    return pairs #actually k-1 nieghbours. k including i. but i not in pair calc.

#***********************************************
#============PRETRAINING ANGLES DIFFERENT=======
#***********************************************
def get_Ris(A,k):
    D_A, I_A = get_topk(A,k)
    knn_idx = I_A #include pts themselves
    knn_coord = filter_rows(A,knn_idx)
    Ri = tf.convert_to_tensor(knn_coord,dtype = tf.float32)
    Ri_idx = knn_idx
    return Ri , Ri_idx
def get_Ris_radius(A,radius,pad_coord=0):
    Aaug = np.concatenate((A,np.full(A[0].shape,pad_coord)[None,:]),0) #Add a row with pad coord as a dummy
    radk_dist, radk_idx, freqs, maxk = get_radiusk(A,radius)
    knn_idx = radk_idx #include pts themselves
    knn_coord = filter_rows(Aaug,knn_idx)
    Ri = knn_coord
    Ri_idx = knn_idx
    return Ri , Ri_idx
def get_radiusk(X,radius,pad = -1):
    PD_X = pairwise_dist(X) #Pairwise distances
    radk_dist = PD_X[PD_X<radius]
    radk_idx = np.argwhere(PD_X<radius)
    radk = np.concatenate((radk_idx,radk_dist[:,None]),1)
    radk = radk[np.lexsort((radk[:,2], radk[:,0]))]
    radk_dist, radk_idx = radk[:,-1],radk[:,:-1]
    #Count n values for each point (idx)
    (unique, counts) = np.unique(radk_idx[:,0], return_counts=True)
    freqs = np.asarray((unique, counts)).T.astype(int)
    maxk = int(np.max(freqs[:,1]))
    #Convert
    radk_dist_sliced = slice_list(radk_dist,(radk_dist==0))
    radk_dist_arr = conv_to_arr_none(radk_dist_sliced,maxk,pad=pad)
    radk_idx_sliced = slice_list(list(radk_idx[:,1]),(radk_dist==0))
    radk_idx_arr = conv_to_arr_none(radk_idx_sliced,maxk,pad=pad).astype(int)
    return radk_dist_arr, radk_idx_arr, freqs, maxk

def get_freqs(Ri_idx,pad = -1):
    freqs = np.count_nonzero(Ri_idx != pad, axis=1)
    idx = np.arange(0,len(freqs))
    freqs = np.concatenate((idx[:,None],freqs[:,None]),1)
    return freqs

def get_DXY_angles_3radius(X,Y,r1,r2,r3,bplot = False):
  #Get Distance matrix
  feat_X = get_X_feat_3radius(X,r1,r2,r3)
  feat_Y = get_X_feat_3radius(Y,r1,r2,r3)
  dist_feat_XY = np.linalg.norm(feat_X[:, None] - feat_Y[None, :],axis=(2))
  if bplot:
      plt.figure()
      plt.imshow(dist_feat_XY)
      plt.colorbar()
      plt.show()
  return dist_feat_XY    
    
def get_X_feat_3radius(A,r1,r2,r3):#UNCHANGED except get_features/pairs_radius
    #TODO - Could be more efficient avoiding to loop 3 times
    X1 = get_X_feat_angles(A,radius = r1)
    X2 = get_X_feat_angles(A,radius = r2)
    X3 = get_X_feat_angles(A,radius = r3)
    return np.concatenate((X1,X2,X3),axis = 1)


#***********************************************
#============PRETRAINING DISTANCE================
#***********************************************
def get_DXY_curves(Y,fX,k,beta,bplot = False,quick = True):
    Y_areas,Y_grad_areas = get_system_curve_areas(Y,k,quick = quick)
    fX_areas,fX_grad_areas = get_system_curve_areas(fX,k,quick = quick)
    dist_curves = tf.math.abs(Y_areas[:,None]-fX_areas[None,:]) + beta * tf.math.abs(Y_grad_areas[:,None]-fX_grad_areas[None,:])
    if bplot:
        plt.figure()
        plt.imshow(dist_curves)
        plt.colorbar()
        plt.show()
    return dist_curves
def get_system_curve_areas(X,k,quick = True):
    x_min,x_max = 1,k
    D_X, I_X = get_topk(X,k)#get_mat_knn_dist(X,k)#
    splines_x = fit_B_spline_mat(D_X) 
    areas, grad_areas = [], []
    for spl in splines_x:
        grad_spl = spl.derivative(nu=1)
        if not quick:
            area,_ = quad(spl, x_min,x_max)
            grad_area,_ = quad(grad_spl, x_min,x_max)
        else:
            area = get_area(spl, x_min,x_max)
            grad_area = get_area(grad_spl, x_min,x_max)
        areas.append(area)
        grad_areas.append(grad_area)
    tf_areas = tf.convert_to_tensor(areas,dtype = np.float32)
    tf_grad_areas = tf.convert_to_tensor(grad_areas,dtype = np.float32)
    return tf_areas, tf_grad_areas
#Fit B-Spline
def fit_B_spline_vec(vec, bplot = False):
    n = len(vec)
    x_rng = range(1,n+1)
    t, c, k = interpolate.splrep(x_rng, vec, s=0, k=4)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    if bplot:
        xx = np.linspace(min(x_rng),max(x_rng),100)
        plt.plot(x_rng, vec, 'bo', label='Original points')
        plt.plot(xx, spline(xx), 'r', label='BSpline')
        plt.grid()
        plt.legend(loc='best')
        plt.show() 
    return spline
def fit_B_spline_mat(D_X):
    splines = []
    for knn_dist in D_X: #each row = 1 datapoint
        spline = fit_B_spline_vec(knn_dist)
        splines.append(spline)
    return splines
def get_area(spl, x_min,x_max):
    #Gets the area under curve by averaging samples
    xx = np.linspace(x_min,x_max,1000)
    vals = spl(xx)
    area = np.mean(vals)*(x_max-x_min)
    return area

#*************************************************************************
#===CODE TO GET BEST K OR RAD ANGLES (FUNC ADAPTED TO BOTH===============
#*************************************************************************
#=============BEST ONLY K OR RAD - NCORR FIXED
def get_best_krad_param_angles(n_iter,Ncorr,
                               n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,
                               k_test = None,rad_test = None):
    #Test either k or rad
    all_dicts = []
    for i_iter in range(n_iter):
        sd = np.random.randint(0,10e5)
        A,B,X_A,X_B,B_shuff,B_idx_map = emb.get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=False)
        dict_run = get_best_krad_run_angles(i_iter,Ncorr,A,B_shuff,B_idx_map,k_test = k_test,rad_test = rad_test)
        all_dicts.append(dict_run)
    #Get all_accs_per_rad
    all_accs_per_p = []
    p_test = k_test if k_test is not None else rad_test
    for p in p_test:
        arr_p = np.array([d[p] for d in all_dicts]) #all values for each k
        all_accs_per_p.append(arr_p)
    #Get best average
    avgs = np.array([np.mean(np.array(accs)) for accs in all_accs_per_p])
    best_idx = np.argmax(avgs)
    best_p = p_test[best_idx] #rad value with highest average
    best_avg = np.mean(np.array(all_accs_per_p[best_idx])) #highest average
    clear_output(wait=True)
    return best_p, best_avg , all_accs_per_p
def get_best_krad_run_angles(i_iter,Ncorr,X,Y,y_idx_map,k_test = None,rad_test = None):
    dict_run = {}
    p_test = k_test if k_test is not None else rad_test
    for p in p_test:
        #jupyter print
        clear_output(wait=True)
        display("Iter %d - Testing param : %.2f ..."%(i_iter,p))
        time.sleep(0.01)
        #code
        if k_test is not None:
            D_XY = get_DXY_angles(X,Y,k=p,bplot = False,bPrint = False)
        elif rad_test is not None:
            D_XY = get_DXY_angles(X,Y,radius=p,bplot = False,bPrint = False)
        X_pretrain,Y_pretrain, matches = get_corres_data(X,Y,D_XY,Ncorr)
        acc , mat = check_corresp(matches,y_idx_map,bPrint = False)
        dict_run[p] = acc
    return dict_run
#=============BEST NCORR OR K RAD
def get_best_Ncorr_krad_run_angles(i_iter,X,Y,y_idx_map,Ncorr_test,k_test = None,rad_test = None):
    dict_run = {}
    p_test = k_test if k_test is not None else rad_test
    pairs = list(itertools.product(Ncorr_test,p_test))
    for Ncorr,p in pairs:
        clear_output(wait=True)
        display("Iter %d - Testing Ncorr : %d - param : %.2f ..."%(i_iter,Ncorr,p))
        time.sleep(0.01)
        if k_test is not None:
            D_XY = get_DXY_angles(X,Y,k=p,bplot = False,bPrint = False)
        elif rad_test is not None:
            D_XY = get_DXY_angles(X,Y,radius=p,bplot = False,bPrint = False)
        X_pretrain,Y_pretrain, matches = get_corres_data(X,Y,D_XY,Ncorr)
        acc , mat = check_corresp(matches,y_idx_map,bPrint = False)
        dict_run[(Ncorr,p)] = acc
    return dict_run
def get_best_Ncorr_krad_params_dist(n_iter,
                                n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,
                                Ncorr_test,k_test = None,rad_test = None):
    all_dicts = []
    p_test = k_test if k_test is not None else rad_test
    pairs = list(itertools.product(Ncorr_test,p_test))
    for i_iter in range(n_iter):
        sd = np.random.randint(0,10e5)
        A,B,X_A,X_B,B_shuff,B_idx_map = emb.get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=False)
        dict_run = get_best_Ncorr_krad_run_angles(i_iter,A,B_shuff,B_idx_map,Ncorr_test,k_test = k_test,rad_test = rad_test)
        all_dicts.append(dict_run)  
    #Get all_accs_per_Ncorr_krad
    all_accs_per_Ncorr_krad = []
    for Ncorr,p in pairs:
        arr_Ncorr_krad = np.array([d[(Ncorr,p)] for d in all_dicts]) #all values for each k
        all_accs_per_Ncorr_krad.append(arr_Ncorr_krad)
    #Get best average
    avgs = np.array([np.mean(np.array(accs)) for accs in all_accs_per_Ncorr_krad])
    best_idx = np.argmax(avgs)
    best_Ncorr_krad = pairs[best_idx] #rad value with highest average
    best_avg = np.mean(np.array(all_accs_per_Ncorr_krad[best_idx])) #highest average
    clear_output(wait=True)
    return best_Ncorr_krad, best_avg , all_accs_per_Ncorr_krad
    

#*****************************************
#===CODE TO GET BEST K BETA Ncorr - DISTANCE====
#***************************************** 
def make_df_pair_params(pairs,all_accs_per_pair_params,p1_name='p1',p2_name='p2'):
    dict_list = []
    for i,(p1,p2) in enumerate(pairs):
        vals = all_accs_per_pair_params[i]
        for val in vals:
            dict_list.append({p1_name:p1,p2_name:p2,'acc':val})
    df = pd.DataFrame(dict_list)
    return df
#=====NCORR AND K=====
def get_best_Ncorr_k_run_dist(i_iter,Ncorr_test,k_test,beta,X,Y,y_idx_map):
    dict_run = {}
    pairs = list(itertools.product(Ncorr_test,k_test))
    for Ncorr,k in pairs:
        clear_output(wait=True)
        display("Iter %d - Testing Ncorr : %d - k : %d ..."%(i_iter,Ncorr,k))
        time.sleep(0.01)
        D_XY = get_DXY_curves(X,Y,k,beta,bplot = False)
        X_pretrain,Y_pretrain, matches = get_corres_data(X,Y,D_XY,Ncorr)
        acc , mat = check_corresp(matches,y_idx_map,bPrint = False)
        dict_run[(Ncorr,k)] = acc
    return dict_run
def get_best_Ncorr_k_params_dist(n_iter,Ncorr_test,k_test,beta,
                        n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd):
    all_dicts = []
    pairs = list(itertools.product(Ncorr_test,k_test))
    for i_iter in range(n_iter):
        sd = np.random.randint(0,10e5)
        A,B,X_A,X_B,B_shuff,B_idx_map = emb.get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=False)
        dict_run = get_best_Ncorr_k_run_dist(i_iter,Ncorr_test,k_test,beta,A,B_shuff,B_idx_map)
        all_dicts.append(dict_run)  
    #Get all_accs_per_k
    all_accs_per_Ncorr_k = []
    for Ncorr,k in pairs:
        arr_Ncorr_k = np.array([d[(Ncorr,k)] for d in all_dicts]) #all values for each k
        all_accs_per_Ncorr_k.append(arr_Ncorr_k)
    #Get best average
    avgs = np.array([np.mean(np.array(accs)) for accs in all_accs_per_Ncorr_k])
    best_idx = np.argmax(avgs)
    best_Ncorr_k = pairs[best_idx] #rad value with highest average
    best_avg = np.mean(np.array(all_accs_per_Ncorr_k[best_idx])) #highest average
    clear_output(wait=True)
    return best_Ncorr_k, best_avg , all_accs_per_Ncorr_k
    

#=====BETA AND K=====
def get_best_beta_k_run_dist(i_iter,Ncorr,beta_test,k_test,X,Y,y_idx_map):
    dict_run = {}
    pairs = list(itertools.product(beta_test,k_test))
    for beta,k in pairs:
        clear_output(wait=True)
        display("Iter %d - Testing beta : %.3f - k : %d ..."%(i_iter,beta,k))
        time.sleep(0.01)
        D_XY = get_DXY_curves(X,Y,k,beta,bplot = False)
        X_pretrain,Y_pretrain, matches = get_corres_data(X,Y,D_XY,Ncorr)
        acc , mat = check_corresp(matches,y_idx_map,bPrint = False)
        dict_run[(beta,k)] = acc
    return dict_run
def get_best_beta_k_params_dist(n_iter,beta_test,k_test,Ncorr,
                        n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd):
    all_dicts = []
    pairs = list(itertools.product(beta_test,k_test))
    for i_iter in range(n_iter):
        sd = np.random.randint(0,10e5)
        A,B,X_A,X_B,B_shuff,B_idx_map = emb.get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=False)
        dict_run = get_best_beta_k_run_dist(i_iter,Ncorr,beta_test,k_test,A,B_shuff,B_idx_map)
        all_dicts.append(dict_run)  
    #Get all_accs_per_k
    all_accs_per_beta_k = []
    for beta,k in pairs:
        arr_beta_k = np.array([d[(beta,k)] for d in all_dicts]) #all values for each k
        all_accs_per_beta_k.append(arr_beta_k)
    #Get best average
    avgs = np.array([np.mean(np.array(accs)) for accs in all_accs_per_beta_k])
    best_idx = np.argmax(avgs)
    best_beta_k = pairs[best_idx] #rad value with highest average
    best_avg = np.mean(np.array(all_accs_per_beta_k[best_idx])) #highest average
    clear_output(wait=True)
    return best_beta_k, best_avg , all_accs_per_beta_k

#=====K ONLY=====
def get_best_k_run_dist(k_test,beta,X,Y,y_idx_map,Ncorr,i_iter):
    dict_run = {}
    for k in k_test:
        #jupyter print
        clear_output(wait=True)
        display("Iter %d - Testing k : %d ..."%(i_iter,k))
        time.sleep(0.01)
        #code
        D_XY = get_DXY_curves(X,Y,k,beta,bplot = False)
        X_pretrain,Y_pretrain, matches = get_corres_data(X,Y,D_XY,Ncorr)
        acc , mat = check_corresp(matches,y_idx_map,bPrint = False)
        dict_run[k] = acc
    return dict_run
def get_best_k_params_dist(n_iter,k_test,beta,Ncorr,
                        n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd):
    all_dicts = []
    for i_iter in range(n_iter):
        sd = np.random.randint(0,10e5)
        A,B,X_A,X_B,B_shuff,B_idx_map = emb.get_data(n_systems,n_concepts,emb_dim,noise,n_epicentres,linearsep,sd,bplot=False)
        dict_run = get_best_k_run_dist(k_test,beta,A,B_shuff,B_idx_map,Ncorr,i_iter)
        all_dicts.append(dict_run)  
    #Get all_accs_per_k
    all_accs_per_k = []
    for k in k_test:
        arr_k = np.array([d[k] for d in all_dicts]) #all values for each k
        all_accs_per_k.append(arr_k)
    #Get best average
    avgs = np.array([np.mean(np.array(accs)) for accs in all_accs_per_k])
    best_idx = np.argmax(avgs)
    best_k_avg = k_test[best_idx] #rad value with highest average
    best_avg = np.mean(np.array(all_accs_per_k[best_idx])) #highest average
    clear_output(wait=True)
    return best_k_avg, best_avg , all_accs_per_k


#*****************************************
#========HELPER FUNCTIONS=================
#*****************************************
def replace_pad_zeros(X,vects_to_replace,maxk):
    #val_to_replace : 1 vect per concept (N,vec)
    N = X.shape[0]
    emb_dim = X.shape[-1]
    for i in range(N):
        for j in range(maxk):
            if np.array_equal(X[i,j,:],vects_to_replace[i]):
                X[i,j,:] = [0] * emb_dim
    return X
def conv_to_arr_none(arr,maxk,pad=None):
    arr = np.array([xi+[pad]*(maxk-len(xi)) for xi in arr])
    return arr
def slice_list(li,cond):
    li = np.split(li, np.where(cond)[0])[1:]
    li = [list(a) for a in li]
    return li
def make_one_hot(idxes,max_idx):
    one_hot = np.zeros((idxes.size,  max_idx)) #np.zeros((one_hot_idx.size, one_hot_idx.max()+1))
    one_hot[np.arange(idxes.size),idxes] = 1
    return one_hot
def filter_rows(X,idx_rows):
    "Filter data compatible with tf and np"
    if tf.is_tensor(X):
        X_filt = tf.gather(X, idx_rows)
    else:
        X_filt = X[idx_rows,:]
    return X_filt
def pairwise_dist(X):
    PD_X = distance.cdist(X,X) #np.linalg.norm(X[:, None] - X[None, :],axis=(2))
#     if not tf.is_tensor(X):
#         X = tf.convert_to_tensor(X,dtype = tf.float32)
#     r = tf.reduce_sum(X*X, 1)
#     r = tf.reshape(r, [-1, 1])
#     PD_X = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r) #Pairwise distances
    return PD_X 
def get_topk(X,k):
  "Gets the k NN val and idx in X (N,k)."
  PD_X = pairwise_dist(X) #Pairwise distances
  topk = tf.math.top_k(-PD_X,k) #CHANGED TO -PD
  topk_dist = -topk.values
  topk_idx = topk.indices
  return topk_dist, topk_idx
def smallestN_indices(X,N):
    Xn = X.numpy() if tf.is_tensor(X) else X
    idx = Xn.ravel().argsort()[:N]
    idx = np.stack(np.unravel_index(idx, Xn.shape)).T
    return idx
def get_rows_idx(X,searched_values):
    #X of shape (N,D). searched_values (M,D). We want the index of where each rows of searched_values are in X
    #code from https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-several-values-in-a-numpy-array
    dims = X.max(0)+1
    X1D = np.ravel_multi_index(X.T,dims)
    searched_valuesID = np.ravel_multi_index(searched_values.T,dims)
    sidx = X1D.argsort()
    searched_idx = sidx[np.searchsorted(X1D,searched_valuesID,sorter=sidx)]
    return searched_idx

#===BEFORE (TESTS DONE WITH THAT)====
    

def get_mat_dist_X_Y(Y,fX,k,beta,bplot = False,quick = True):
    Y_areas,Y_grad_areas = get_system_curve_areas2(Y,k,quick = quick)
    fX_areas,fX_grad_areas = get_system_curve_areas2(fX,k,quick = quick)
    dist_curves = tf.math.abs(Y_areas[:,None]-fX_areas[None,:]) + beta * tf.math.abs(Y_grad_areas[:,None]-fX_grad_areas[None,:])
    if bplot:
        plt.figure()
        plt.imshow(dist_curves)
        plt.colorbar()
        plt.show()
    return dist_curves
def get_system_curve_areas2(X,k,quick = True):
    x_min,x_max = 1,k
    D_X, I_X = get_mat_knn_dist2(X,k)# 
    splines_x = fit_B_spline_mat(D_X) 
    areas, grad_areas = [], []
    for spl in splines_x:
        grad_spl = spl.derivative(nu=1)
        if not quick:
            area,_ = quad(spl, x_min,x_max)
            grad_area,_ = quad(grad_spl, x_min,x_max)
        else:
            area = get_area(spl, x_min,x_max)
            grad_area = get_area(grad_spl, x_min,x_max)
        areas.append(area)
        grad_areas.append(grad_area)
    tf_areas = tf.convert_to_tensor(areas,dtype = np.float32)
    tf_grad_areas = tf.convert_to_tensor(grad_areas,dtype = np.float32)
    return tf_areas, tf_grad_areas
def get_mat_knn_dist2(X,k):
    knn_dist_list, knn_idx_list = [], []
    for x in X:
        x = x[None,:]
        knn_idx, knn_dist = get_knn_idx2(x,X,k)
        knn_dist_list.append(knn_dist)
        knn_idx_list.append(knn_idx)
    D_X = np.array(knn_dist_list)
    I_X = np.array(knn_idx_list)
    return D_X , I_X
def get_mat_knn_idx2(X,k):
    knn_idx_list = []
    for x in X:
        x = x[None,:]
        knn_idx, knn_dist = get_knn_idx2(x,X,k)
        knn_idx_list.append(knn_idx)
    I_X = np.array(knn_idx_list)
    return I_X
def get_knn_idx2(x,A,k):
    n_concepts = A.shape[-2]
    x_dist = list(distance.cdist(x, A, 'euclidean').squeeze())
    idx = range(n_concepts)
    sorted_x_dist, sorted_idx = (list(t) for t in zip(*sorted(zip(x_dist, idx))))
    knn_idx = sorted_idx[1:k+1]
    knn_dist = sorted_x_dist[1:k+1]
    return knn_idx, knn_dist

  