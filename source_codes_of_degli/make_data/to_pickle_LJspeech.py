# -*- coding: utf-8 -*-
###############################################################################
import glob
import numpy as np
import pickle
import soundfile as sf
np.random.seed(0)

from audio_sample_config import sample_list


## Data #######################################################################
wav_dir   = '/raid/home/abrol/Siya20577/Emotional-TTS/resources/LJSpeech/wavs'
wav_files = glob.glob(wav_dir + "/" + "*.wav")
Num_data  = len(wav_files)
print('Num_data:', str(Num_data))

sample_list = [wav_dir + "/" + fname for fname in sample_list]
wav_files_ = list(set(wav_files) - set(sample_list))
wav_files_ = list(np.random.permutation(wav_files_))


## Parametrs ##################################################################
fs = 22050
winlen = 1024
shift  = 256
lf = 24064

num_train = 12500
num_valid = 300
num_test = 300


## Splitting ##################################################################
train_fns = wav_files_[:num_train]
valid_fns = wav_files_[num_train:num_train+num_valid]
test_fns  = wav_files_[num_train+num_valid:]
test_fns.extend(sample_list)

train_speeches = []
valid_speeches = []
test_speeches = []


## Training data ##############################################################
iter_ind = 0

for fn in train_fns:
    speech_tmp, _ = sf.read(fn)
    num_iter = int(np.floor(len(speech_tmp)/lf))
    print('1', num_iter, len(speech_tmp), lf)
    
    for i in range(num_iter):
        speech_ttmp = speech_tmp[i*lf:(i+1)*lf]
        speech_ttmp = np.concatenate((np.zeros(winlen-shift,),speech_ttmp,
                                          np.zeros(winlen-shift,)), axis=0)
        speech_ttmp /= np.max(np.abs(speech_ttmp))
        print('speech_ttmp', speech_ttmp.shape)
        train_speeches.append(speech_ttmp.astype(np.float32))
        
    print(str(iter_ind)+'/'+str(Num_data))
    iter_ind += 1

                             
## Validation data ############################################################
for fn in valid_fns:

    speech_tmp, _ = sf.read(fn)
    num_iter = int(np.floor(len(speech_tmp)/lf))                         
    
    for i in range(num_iter):
        speech_ttmp = speech_tmp[i*lf:(i+1)*lf]
        speech_ttmp = np.concatenate((np.zeros(winlen-shift,),speech_ttmp,
                                          np.zeros(winlen-shift,)), axis=0)
        speech_ttmp /= np.max(np.abs(speech_ttmp))
        valid_speeches.append(speech_ttmp.astype(np.float32))

    print(str(iter_ind)+'/'+str(Num_data))
    iter_ind += 1

                             
## Test data ##################################################################
for fn in test_fns:
    speech_ttemp, _ = sf.read(fn)
    speech_ttemp /= np.max(np.abs(speech_ttemp))
    test_speeches.append(speech_ttemp.astype(np.float32))
    
    print(str(iter_ind)+'/'+str(Num_data))
    iter_ind += 1


## Export #####################################################################
with open('../pickles/train_speeches_LJ.pickle', mode='wb') as train_speeches_pickle:
    pickle.dump(train_speeches, train_speeches_pickle, protocol=4)
    
with open('../pickles/valid_speeches_LJ.pickle', mode='wb') as valid_speeches_pickle:
    pickle.dump(valid_speeches, valid_speeches_pickle, protocol=4)
    
with open('../pickles/test_speeches_LJ.pickle', mode='wb') as test_speeches_pickle:
    pickle.dump(test_speeches, test_speeches_pickle, protocol=4)