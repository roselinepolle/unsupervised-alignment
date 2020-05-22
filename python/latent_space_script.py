import numpy as np 
import pandas as pd
import math
import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import time
import utils_upload as utils
import aligners_upload as aligners
import embeddings_upload as embeddings

timestr = time.strftime("%Y%m%d-%H%M%S")

def define_train_batch_step():
    @tf.function
    def train_batch_step(A_batch, B_batch, C_batch):
        """Training function for batch step (pointwise loss)"""

        with tf.GradientTape(persistent=True) as tape_pointwise:

            # Encode all with their own encoders
            encoded_A = A_encoder(A_batch, training=True)
            encoded_B = B_encoder(B_batch, training=True)
            encoded_C = C_encoder(C_batch, training=True)

            # Decode all with their own decoders
            decoded_A = A_decoder(encoded_A, training=True)
            decoded_B = B_decoder(encoded_B, training=True)
            decoded_C = C_decoder(encoded_C, training=True)

            # Self loss terms
            self_lossA = aligners.flex_cycle_loss(
                A_batch, gf_x=decoded_A, one_system=True
                )
            self_lossB = aligners.flex_cycle_loss(
                B_batch, gf_x=decoded_B, one_system=True
                )
            self_lossC = aligners.flex_cycle_loss(
                C_batch, gf_x=decoded_C, one_system=True
                )
                
            # Decode each with another decoder
            atob = B_decoder(encoded_A, training=True)
            btoc = C_decoder(encoded_B, training=True)
            ctoa = A_decoder(encoded_C, training=True)
            atoc = C_decoder(encoded_A, training=True)
            btoa = A_decoder(encoded_B, training=True)
            ctob = B_decoder(encoded_C, training=True)

            # Revert this rep back to original
            encode_revertab = B_encoder(atob, training=True)
            encode_revertbc = C_encoder(btoc, training=True)
            encode_revertca = A_encoder(ctoa, training=True)
            encode_revertac = A_encoder(atoc, training=True)
            encode_revertba = B_encoder(btoa, training=True)
            encode_revertcb = C_encoder(ctob, training=True)

            decode_revertab = A_decoder(encode_revertab, training=True)
            decode_revertbc = B_decoder(encode_revertbc, training=True)
            decode_revertca = C_decoder(encode_revertca, training=True)
            decode_revertac = A_decoder(encode_revertac, training=True)
            decode_revertba = B_decoder(encode_revertba, training=True)
            decode_revertcb = C_decoder(encode_revertcb, training=True)

            # Calculate loss terms from error in cycle via other system
            loss_ab = aligners.flex_cycle_loss(
                A_batch, gf_x=decode_revertab, one_system=True
                )
            loss_ac = aligners.flex_cycle_loss(
                A_batch, gf_x=decode_revertac, one_system=True
                )
            loss_1 = loss_ab + loss_ac
            
            loss_bc = aligners.flex_cycle_loss(
                B_batch, gf_x=decode_revertbc, one_system=True
                    )
            loss_ba = aligners.flex_cycle_loss(
                B_batch, gf_x=decode_revertba, one_system=True
                )
            loss_2 = loss_ba + loss_bc
            
            loss_ca = aligners.flex_cycle_loss(
                C_batch, gf_x=decode_revertca, one_system=True
                )
            loss_cb = aligners.flex_cycle_loss(
                C_batch, gf_x=decode_revertcb, one_system=True
                )
            loss_3 = loss_ca + loss_cb

            self_loss = self_lossA + self_lossB + self_lossC
            self_loss_list = [self_lossA, self_lossB, self_lossC]
            cycle_loss = loss_1 + loss_2 + loss_3
            cycle_loss_list = [loss_ab,
                                  loss_ac,
                                  loss_ba,
                                  loss_bc,
                                  loss_ca,
                                  loss_cb]
            pointwise_loss = (self_loss + cycle_loss) 

        # gradients calculated based on the tape above
        gradients_a = tape_pointwise.gradient(
            pointwise_loss, A_encoder.trainable_variables
            )
        gradients_b = tape_pointwise.gradient(
            pointwise_loss, B_encoder.trainable_variables
            )
        gradients_c = tape_pointwise.gradient(
            pointwise_loss, C_encoder.trainable_variables
            )

        optimizer_encodea.apply_gradients(zip(
            gradients_a, A_encoder.trainable_variables
            ))
        optimizer_encodeb.apply_gradients(zip(
            gradients_b, B_encoder.trainable_variables
            ))
        optimizer_encodec.apply_gradients(zip(
            gradients_c, C_encoder.trainable_variables
            ))
                        

        # Gradients for decoders:
        gradients_da = tape_pointwise.gradient(
            cycle_loss, A_decoder.trainable_variables
            )
        gradients_db = tape_pointwise.gradient(
            cycle_loss, B_decoder.trainable_variables
            )
        gradients_dc = tape_pointwise.gradient(
            cycle_loss, C_decoder.trainable_variables
            )

        optimizer_decodea.apply_gradients(
            zip(gradients_da, A_decoder.trainable_variables
            ))
        optimizer_decodeb.apply_gradients(
            zip(gradients_db, B_decoder.trainable_variables
            ))
        optimizer_decodec.apply_gradients(
            zip(gradients_dc, C_decoder.trainable_variables
            ))

        return pointwise_loss, self_loss, cycle_loss, self_loss_list, cycle_loss_list
    return train_batch_step

def define_train_full_step():
    @tf.function
    def train_full_step(A_all, B_all, C_all, gmm_kernel, 
                            dist_loss_z, dist_loss_final,
                            rawgmm_A, rawgmm_B, rawgmm_C):

        """
        Training function for full step (distribution loss)
        """

        with tf.GradientTape(persistent=True) as tape_nll:
                        
            # Get latent space reps of each system

            latent_A = A_encoder(A_all, training=True)
            latent_B = B_encoder(B_all, training=True)
            latent_C = C_encoder(C_all, training = True)

            # Dist versions 
            latent_gmm_a = aligners.assemble_gmm(
                    latent_A, batches=True, kernel_size=gmm_kernel
                )
            latent_gmm_b = aligners.assemble_gmm(
                    latent_B, batches=True, kernel_size=gmm_kernel
                )
            latent_gmm_c = aligners.assemble_gmm(
                    latent_C, batches=True, kernel_size=gmm_kernel
                )

            # Calculate maxloglik for each
            maxlika_z = utils.loglik(latent_gmm_a, latent_A)
            maxlikb_z = utils.loglik(latent_gmm_b, latent_B)
            maxlikc_z = utils.loglik(latent_gmm_c, latent_C)

            # loglik of sample from one other system's latent rep
            loglik_BA_z = tf.minimum(
                utils.loglik(latent_gmm_a, latent_B), maxlika_z
                )
            loglik_CB_z = tf.minimum(
                utils.loglik(latent_gmm_b, latent_C), maxlikb_z
                )
            loglik_AC_z = tf.minimum(
                utils.loglik(latent_gmm_c, latent_A), maxlikc_z
                )

            # loglik of cross-mapping as sample from original gmm
            maxlika = utils.loglik(rawgmm_A, A_all)
            maxlikb = utils.loglik(rawgmm_B, B_all)
            maxlikc = utils.loglik(rawgmm_C, C_all)

            loglik_BA = tf.minimum(
                utils.loglik(rawgmm_B, B_decoder(latent_A)), maxlikb
                )
            loglik_CB = tf.minimum(
                utils.loglik(rawgmm_C, C_decoder(latent_B)), maxlikc
                )
            loglik_AC = tf.minimum(
                utils.loglik(rawgmm_A, A_decoder(latent_C)), maxlika
                )
            loglik_BC = tf.minimum(
                utils.loglik(rawgmm_B, B_decoder(latent_C)), maxlikb
                )
            loglik_CA = tf.minimum(
                utils.loglik(rawgmm_C, C_decoder(latent_A)), maxlikc
                )
            loglik_AB = tf.minimum(
                utils.loglik(rawgmm_A, A_decoder(latent_B)), maxlika
                )

            latent_dist_loss = (
                -loglik_BA_z-loglik_CB_z-loglik_AC_z
                )/3*n_concepts

            final_dl_BA = (-loglik_BA)/n_concepts
            final_dl_CB = (-loglik_CB)/n_concepts
            final_dl_AC = (-loglik_AC)/n_concepts
            final_dl_BC = (-loglik_BC)/n_concepts
            final_dl_CA = (-loglik_CA)/n_concepts
            final_dl_AB = (-loglik_AB)/n_concepts
            final_dl_list = [final_dl_AB, 
                            final_dl_AC,
                            final_dl_BA, 
                            final_dl_BC, 
                            final_dl_CA, 
                            final_dl_CB]

            dist_loss = dist_c_z * latent_dist_loss 
            + dist_c_final *(final_dl_BA 
                              + final_dl_CB 
                              + final_dl_AC
                              + final_dl_BC
                              + final_dl_CA
                              + final_dl_AB)
                            

        # gradients calculated based on the tape above

        gradients_a = tape_nll.gradient(dist_loss, A_encoder.trainable_variables)
        gradients_b = tape_nll.gradient(dist_loss, B_encoder.trainable_variables)
        gradients_c = tape_nll.gradient(dist_loss, C_encoder.trainable_variables)

        optimizer_encode_fulla.apply_gradients(zip(
            gradients_a, A_encoder.trainable_variables
            ))
        optimizer_encode_fullb.apply_gradients(zip(
            gradients_b, B_encoder.trainable_variables
            ))
        optimizer_encode_fullc.apply_gradients(zip(
            gradients_c, C_encoder.trainable_variables
            ))

        if dist_loss_final != 0:
            gradients_da = tape_nll.gradient(dist_loss, A_decoder.trainable_variables)
            gradients_db = tape_nll.gradient(dist_loss, B_decoder.trainable_variables)
            gradients_dc = tape_nll.gradient(dist_loss, C_decoder.trainable_variables)

            optimizer_decode_fulla.apply_gradients(zip(
                gradients_da, A_decoder.trainable_variables
                ))
            optimizer_decode_fullb.apply_gradients(zip(
                gradients_db, B_decoder.trainable_variables
                ))
            optimizer_decode_fullc.apply_gradients(zip(
                gradients_dc, C_decoder.trainable_variables
                ))

        return dist_loss, latent_dist_loss, final_dl_list
    return train_full_step


## Training params
pretrain_epochs = 10 # No. epochs training on cycle loss alone 
EPOCHS = 500-pretrain_epochs
subrestart_every = 200 # subrestart every __ epochs
RESTARTS = 100
dist_c_z = 0.000005 # weight of latent distribution loss v. cycle in loss
dist_c_final = 0 # weight of output distribution loss v. cycle in loss

# Dimensionalities to test
dim_list = [2, 3]

# Experiment loop
for n_dim in dim_list:
    
    loss_collection_train = []

    ## Define systems
    n_systems = 3
    n_concepts = 200
    n_epicentres = 5 # Number of concept "clumps"
    epicentre_range = 5 # Spread of centres about 0 (from -range to + range)
    gaussian_sigma = 0.8 # Variance for each concept epicentre
    noise = 0.01 # SD of noise kernel
    gmm_kernel = 0.01 # Kernel bandwidth for GMM based on points
    n_batch = 5 # Batches for cycle loss training each epoch
    batch_size = 75 #

    learning_rate_batch = 0.001
    learning_rate_full = 0.001


    systems, noisy_systems = embeddings.create_n_systems(
                                            n_epicentres=n_epicentres, 
                                            epicentre_range=epicentre_range, 
                                            n_dim=n_dim, 
                                            num_concepts=n_concepts, 
                                            sigma=gaussian_sigma, 
                                            noise_size=noise, 
                                            n_systems=n_systems, 
                                            return_noisy=True, 
                                            plot = False, 
                                            rotation = True
                                            )

    A = systems[0]
    B = systems[1]
    C = systems[2]

    # Find positions relative to the mean in each dimension
    A = A - tf.expand_dims(tf.reduce_mean(A, axis=-2), axis=1)
    B = B - tf.expand_dims(tf.reduce_mean(B, axis=-2), axis=1)
    C = C - tf.expand_dims(tf.reduce_mean(C, axis=-2), axis=1)

    # Create gmms for original systems
    rawgmm_A = aligners.assemble_gmm(
                    A, batches=True, kernel_size=gmm_kernel
                )
    rawgmm_B = aligners.assemble_gmm(
                    B, batches=True, kernel_size=gmm_kernel
                )
    rawgmm_C = aligners.assemble_gmm(
                    C, batches=True, kernel_size=gmm_kernel
                )

    # Define optimisers
    ### Batch step optimisers
    optimizer_encodea= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )
    optimizer_decodea= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )

    optimizer_encodeb= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )
    optimizer_decodeb= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )

    optimizer_encodec= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )
    optimizer_decodec= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_batch
        )

    ### Full step optimisers
    optimizer_encode_fulla= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )
    optimizer_encode_fullb= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )
    optimizer_encode_fullc= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )

    optimizer_decode_fulla= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )
    optimizer_decode_fullb= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )
    optimizer_decode_fullc= tf.keras.optimizers.Adam(
        learning_rate=learning_rate_full
        )

    ## Calculate ceiling accuracy for mapping between systems
    AtoB_ceiling = utils.calculate_ceiling(
        noisy_systems[0], noisy_systems[1]
        )
    BtoC_ceiling = utils.calculate_ceiling(
        noisy_systems[1], noisy_systems[2]
    )
    CtoA_ceiling = utils.calculate_ceiling(
        noisy_systems[2], noisy_systems[0]
    )

    # Define dimensionality of latent space and hidden layers
    n_dim_latent_in = n_dim
    n_dim_hidden_layers = max(n_dim, 10)

    # Initialise best output distribution loss
    best_dist_loss = 1000000000

    for restart in range(RESTARTS):

        # Schedule for p(choose_best_weights)
        if restart < 10:
            p_thresh = 0
        elif restart < 70:
            p_thresh = 0.5
        else:
            p_thresh = 0.9
        
        # For each encoder/decoder pair 
        # Sample from random uniform to determine if best weights are used 
        # Else, instantiate encoder/decoder pair from scratch
        A_random = np.random.uniform()
        if A_random < p_thresh and restart > 0:
            A_encoder.set_weights(A_encoder_bw)
            A_decoder.set_weights(A_decoder_bw)
        else:
            A_encoder = aligners.Encoder(
            hidden = n_dim_hidden_layers, 
            n_dim_encode=n_dim_latent_in
            )
            A_decoder = aligners.Decoder(
            hidden = n_dim_hidden_layers, 
            n_dim_out=n_dim
            )
        
        B_random = np.random.uniform()
        if B_random < p_thresh and restart > 0:
            B_encoder.set_weights(B_encoder_bw)
            B_decoder.set_weights(B_decoder_bw)
        else:
            B_encoder = aligners.Encoder(
            hidden = n_dim_hidden_layers, 
            n_dim_encode=n_dim_latent_in
            )
            B_decoder = aligners.Decoder(
            hidden = n_dim_hidden_layers, 
            n_dim_out=n_dim
            )

        C_random = np.random.uniform()
        if C_random < p_thresh and restart > 0:
            C_encoder.set_weights(C_encoder_bw)
            C_decoder.set_weights(C_decoder_bw)
        else:
            C_encoder = aligners.Encoder(
            hidden = n_dim_hidden_layers, 
            n_dim_encode=n_dim_latent_in
            )
            C_decoder = aligners.Decoder(
            hidden = n_dim_hidden_layers, 
            n_dim_out=n_dim
            )

        # Re-define training functions to avoid strange tf error
        train_batch_step = define_train_batch_step()
        train_full_step = define_train_full_step()

        # Pretrain using cycle loss only
        for i in range(pretrain_epochs):
            A_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            A_shuff = tf.expand_dims(tf.gather(tf.squeeze(A), A_shuff_idx), axis=0)

            B_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            B_shuff = tf.expand_dims(tf.gather(tf.squeeze(B), B_shuff_idx), axis=0)

            C_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            C_shuff = tf.expand_dims(tf.gather(tf.squeeze(C), C_shuff_idx), axis=0)


            for batch in range(n_batch): 

                concepts = list(range(n_concepts))
                batch_idx = random.sample(concepts, batch_size)

                A_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(A_shuff), batch_idx)), axis=0
                    )
                B_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(B_shuff), batch_idx)), axis=0
                    )
                C_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(C_shuff), batch_idx)), axis=0
                    )

                ___, ___, ___, ___, ___ = train_batch_step(
                                        A_batch, B_batch, C_batch
                                        )
 
        # Full training with cycle loss and distribution loss
        for epoch in range(EPOCHS):
            
            # Shuffle full concept lists
            A_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            A_shuff = tf.expand_dims(
                tf.gather(tf.squeeze(A), A_shuff_idx), axis=0
                )

            B_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            B_shuff = tf.expand_dims(
                    tf.gather(tf.squeeze(B), B_shuff_idx), axis=0
                )
            
            C_shuff_idx = tf.random.shuffle(list(range(n_concepts)))
            C_shuff = tf.expand_dims(
                tf.gather(tf.squeeze(C), C_shuff_idx), axis=0
                )


            for batch in range(n_batch): 
                
                # Randomly select concepts for each batch
                concepts = list(range(n_concepts))
                batch_idx = random.sample(concepts, batch_size)

                A_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(A_shuff), batch_idx)), axis=0
                    )
                B_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(B_shuff), batch_idx)), axis=0
                    )
                C_batch = tf.expand_dims(
                    tf.random.shuffle(
                        tf.gather(tf.squeeze(C_shuff), batch_idx)), axis=0
                    )

                # Batch step (cycle/self loss)
                point_l, self_l, cycle_l, self_l_list, cycle_l_list = train_batch_step(
                                        A_batch, B_batch, C_batch
                                        )

            # Full step (distribution loss)
            dist_l, latent_dist_l, final_dl_list = train_full_step(
                A_shuff, B_shuff, C_shuff, gmm_kernel, 
                dist_c_z, dist_c_final, 
                rawgmm_A, rawgmm_B, rawgmm_C
            )

            # Total output distribution loss (held-out metric)
            final_dist = tf.reduce_sum(
                tf.convert_to_tensor(final_dl_list)
                ).numpy()/len(final_dl_list)

            # Calculate mapping accuracies for this epoch
            BfromA = B_decoder(A_encoder(A))
            acc1BA, acc5BA, acc10BA, acchalfBA = utils.mapping_accuracy(
                tf.squeeze(BfromA), tf.squeeze(B))

            CfromB = C_decoder(B_encoder(B))
            acc1CB, acc5CB, acc10CB, acchalfCB = utils.mapping_accuracy(
                tf.squeeze(CfromB), tf.squeeze(C)
                )

            AfromC = A_decoder(C_encoder(C))
            acc1AC, acc5AC, acc10AC, acchalfAC = utils.mapping_accuracy(
                tf.squeeze(AfromC), tf.squeeze(A)
                )

            # Compile log for dataframe
            loss_collection_train.append([
                        restart,
                        epoch,
                        point_l.numpy(),
                        self_l.numpy(), 
                        cycle_l.numpy(),
                        latent_dist_l.numpy()[0],
                        final_dl_list[0].numpy()[0], final_dl_list[1].numpy()[0], 
                        final_dl_list[2].numpy()[0], final_dl_list[3].numpy()[0], 
                        final_dl_list[4].numpy()[0], final_dl_list[5].numpy()[0],
                        final_dist/6,
                        acc1BA/AtoB_ceiling, 
                        acc5BA, 
                        acc10BA, 
                        acchalfBA,
                        acc1CB/BtoC_ceiling,
                        acc5CB, 
                        acc10CB, 
                        acchalfCB, 
                        acc1AC/CtoA_ceiling,
                        acc5AC, 
                        acc10AC, 
                        acchalfAC
                ])

            template = "restart: {}, t: {}, cycle: {}, dist: {}"

            if epoch % 100 == 0:
                print(template.format(
                    restart, 
                    epoch, 
                    point_l.numpy(), 
                    dist_c_z * latent_dist_l.numpy()[0]
                                ))

            #pre_subs = time.time()
            # Subrestarts based on output distribution loss
            if epoch % subrestart_every == 0 and epoch != 0:
                track_perf = np.zeros([3,3])

                track_perf[0][1] = final_dl_list[0]
                track_perf[0][2] = final_dl_list[1]
                track_perf[1][0] = final_dl_list[2]
                track_perf[1][2] = final_dl_list[3]
                track_perf[2][0] = final_dl_list[4]
                track_perf[2][1] = final_dl_list[5]

                track_perf = tf.convert_to_tensor(track_perf)

                tf.linalg.set_diag(
                    track_perf, np.array(self_l_list)
                    )


                decoder_sums = tf.reduce_sum(track_perf, axis=0)
                encoder_sums = tf.reduce_sum(track_perf, axis=1)
                total_sums = decoder_sums + encoder_sums

                
                # Reset encoder/decoder pair with highest loss
                worst_arg = tf.argmax(total_sums)
                if worst_arg == 0:
                    A_encoder = aligners.Encoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_encode=n_dim_latent_in
                        )
                    A_decoder = aligners.Decoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_out=n_dim
                        )

                    train_batch_step = define_train_batch_step()
                    train_full_step = define_train_full_step()
                    
                if worst_arg == 1:
                    B_encoder = aligners.Encoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_encode=n_dim_latent_in
                        )
                    B_decoder = aligners.Decoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_out=n_dim
                        )
                    
                    train_batch_step = define_train_batch_step()
                    train_full_step = define_train_full_step()
                    
                if worst_arg == 2:

                    C_encoder = aligners.Encoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_encode=n_dim_latent_in
                        )
                    C_decoder = aligners.Decoder(
                        hidden = n_dim_hidden_layers, 
                        n_dim_out=n_dim
                        )           
                    reinstantiation_t = time.time()
                    reinstantiation_duration = reinstantiation_t - sums_t
                   
                    train_batch_step = define_train_batch_step()
                    train_full_step = define_train_full_step()


        # Keep track of weights for all models which minimise output dist loss
        if final_dist < best_dist_loss:
            A_encoder_bw = A_encoder.get_weights()
            A_decoder_bw = A_decoder.get_weights()
            B_encoder_bw = B_encoder.get_weights()
            B_decoder_bw = B_decoder.get_weights()
            C_encoder_bw = C_encoder.get_weights()
            C_decoder_bw = C_decoder.get_weights() 

            best_dist_loss = final_dist


    COLUMNS = [
        "restart",
        "epoch",
        "point_loss",
        "self_loss", 
        "cycle_loss", 
        "latent_dist_loss",
        "dist_loss_AB", "dist_loss_AC", "dist_loss_BA",
        "dist_loss_BC", "dist_loss_CA", "dist_loss_CB",
        "final_dist_loss_mean",
        "acc1BA", "acc5BA", "acc10BA", "acchalfBA",
        "acc1CB", "acc5CB", "acc10CB", "acchalfCB",
        "acc1AC", "acc5AC", "acc10AC", "acchalfAC"
        ]

    df_train = pd.DataFrame(
        loss_collection_train, 
        columns = COLUMNS
        )

    title = "Train_batch_3sys_{}D_{}.csv".format(n_dim, timestr)
    df_train.to_csv(title)


