# -*- coding: utf-8 -*-
###############################################################################
import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from utils.my_models import ComplexGatedConvAmpM
from utils.my_losses import phase_sensitive_l2
from utils.my_others import STFT, iSTFT, DeGLIBlock
tf.reset_default_graph()


## Parameters #################################################################
fs = 22050
winlen = 1024
nfft  = 1024
shift = 256
lf = 25600  # 24064 + (1024-256)*2

num_freq = int(np.floor(nfft/2))+1
num_frame = 97

num_ch = 64
epoch = 300
batch_size = 32
l_rate_init = 4e-4
SNR_MAX = 12.
SNR_MIN = -6.

model_key = 'degli'
model_dir = '../model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
    
## Audio loading ##############################################################
pickle_dir = '../pickles/'
with open(pickle_dir+'train_speeches_LJ.pickle', mode='rb') as train_speeches_pickle:
     train_speeches = np.array(pickle.load(train_speeches_pickle))
       
num_train = len(train_speeches)
for i in range(num_train):
    print('Train speech shape: ' + str(train_speeches[i].shape))


## Placehoders ################################################################
signals = tf.placeholder(tf.float32, (None, lf))
adn_tf  = tf.placeholder(tf.complex64, (None, num_frame, num_freq))
l_rate_tmpin = tf.placeholder(tf.float32)


## Pre-processing #############################################################
stft  = STFT(winlen, shift, nfft)
istft = iSTFT(winlen, shift, nfft)

c_oracle = stft(signals)
amp_ora  = tf.cast(tf.abs(c_oracle), tf.complex64)
adn_tf_sacle = adn_tf*tf.linalg.norm(c_oracle, ord='fro', axis=[-2,-1],
                                     keepdims=True)
c_adwn  = c_oracle + adn_tf_sacle


## Main #######################################################################
model = ComplexGatedConvAmpM(num_ch=num_ch, scope=model_key)
degli_block = DeGLIBlock(amp_ora, stft, istft, model)   
c_est = degli_block(c_adwn)


## Loss #######################################################################
loss = phase_sensitive_l2(c_oracle, c_est)
optimizer = tf.train.AdamOptimizer(l_rate_tmpin)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients = [
    None if gradient is None else tf.clip_by_norm(gradient, 100.0)
    for gradient in gradients]
train_step = optimizer.apply_gradients(zip(gradients, variables))
extra_update_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


## Training ###################################################################
init = tf.global_variables_initializer()
iter_train = int(np.floor(num_train/batch_size))
l_rate_tmp = l_rate_init


## Main loop ##################################################################
print('Training start ...')
with tf.Session() as sess:
    
    sess.run(init)
    saver = tf.train.Saver()
    
    for ii in range(1):

        loss_valid = 0.0
        train_speeches = np.random.permutation(train_speeches)
        print('train_speeches', train_speeches.shape)
        
        ## Training
        print('')
        print('Epoch: ' + str(ii))        
        for i in range(iter_train):
            
            batch_speeches = train_speeches[i*batch_size:i*batch_size+batch_size,:]
            print('batch_speeches', batch_speeches.shape)
            adn = np.random.randn(batch_size, num_frame, num_freq) \
                    + 1j*np.random.randn(batch_size, num_frame, num_freq)
            norm_adn = np.linalg.norm(adn, ord='fro', axis = (1,2))
            adn = adn / norm_adn[:, np.newaxis, np.newaxis] \
                    * 10.**(-1.*(np.random.rand()*(SNR_MAX-SNR_MIN)+SNR_MIN)/20)
            
            sess.run([train_step, extra_update_step], 
                     feed_dict={signals: batch_speeches,
                                adn_tf: adn,
                                l_rate_tmpin: l_rate_tmp
                                }
            )
            
        if (ii+1)%100 == 0:
            l_rate_tmp /= 2.
            
        saver.save(sess, model_dir+'/model.ckpt', global_step=ii)