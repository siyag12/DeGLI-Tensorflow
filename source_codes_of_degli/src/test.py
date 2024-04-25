# -*- coding: utf-8 -*-
###############################################################################
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import soundfile as sf
tf.reset_default_graph()
tf.set_random_seed(0) 

from utils.my_models import ComplexGatedConvAmpM
from utils.my_others import STFT, iSTFT, DeGLIBlock10
from utils.my_module import zero_pad, normalize_1d


## Parameters #################################################################
fs = 22050
winlen = 1024
nfft  = 1024
shift = 256

num_ch = 64
batch_size = 1
save_ind = 299

model_key = 'degli'
model_dir = '../model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
data_dir = '../reconstructed/'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
    
## Audio loading ##############################################################
pickle_dir = '../pickles/'
with open(pickle_dir+'test_speeches_LJ.pickle', mode='rb') as test_speeches_pickle:
     test_speeches = pickle.load(test_speeches_pickle)
     
num_test = len(test_speeches)
    

## Placehoders ################################################################
signals = tf.placeholder(tf.float32, (None, None))


## Pre-processing #############################################################
stft  = STFT(winlen, shift, nfft)
istft = iSTFT(winlen, shift, nfft)

c_oracle = stft(signals)
amp_ora  = tf.cast(tf.abs(c_oracle), tf.complex64)
c_adwn  = amp_ora * tf.exp(1j*tf.complex(tf.random.uniform(tf.shape(amp_ora),
                                         minval=-np.pi, maxval=np.pi),0.))


## Main #######################################################################
model = ComplexGatedConvAmpM(num_ch=num_ch, scope=model_key)
degli_block010 = DeGLIBlock10(amp_ora, stft, istft, model)
c_est010 = degli_block010(c_adwn)
f_est010 = istft(c_est010)


## Training ###################################################################
init = tf.global_variables_initializer()


## Main loop ##################################################################
with tf.Session() as sess:
    
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_dir + 'model.ckpt-' + str(save_ind))
    
    for ii in range(1):        
        # Validation     
        for i in range(num_test):

            speech_temp_ = test_speeches[i]
            lf = len(speech_temp_)
            speech_temp = np.concatenate((np.zeros(winlen-shift,),
                                          speech_temp_ ,
                                          np.zeros(winlen-shift,)), axis=0)
            batch_speeches = zero_pad(speech_temp, winlen, shift)[None,:]
        
            f_est010_np = sess.run(f_est010, feed_dict={signals: batch_speeches})
            
            
            f_est010_np = f_est010_np[0,winlen-shift:winlen-shift+lf]
            f_est010_np = normalize_1d(f_est010_np)            
            sf.write(data_dir + '/' + str(i).zfill(3) + '_degli010.wav',
                     f_est010_np, fs)    