import os
import sys
import pickle
import numpy as np

## Audio loading ##############################################################
pickle_dir = '../pickles/'
with open(pickle_dir+'train_speeches_LJ.pickle', mode='rb') as train_speeches_pickle:
     train_speeches = np.array(pickle.load(train_speeches_pickle))
       
num_train = len(train_speeches)
print('num_train', num_train)
for i in range(num_train):
    print('Train speech shape: ' + str(train_speeches[i].shape))
    break


train_speeches = np.random.permutation(train_speeches)
print('train_speeches', train_speeches.shape)


fs = 22050
winlen = 1024
nfft  = 1024
shift = 256
lf = 25600  # 24064 + (1024-256)*2
batch_size = 32
iter_train = int(np.floor(num_train/batch_size))
num_freq = int(np.floor(nfft/2))+1
num_frame = 97
SNR_MAX = 12.
SNR_MIN = -6.

for i in range(iter_train):
            
    batch_speeches = train_speeches[i*batch_size:i*batch_size+batch_size,:]
    print('batch_speeches', batch_speeches.shape)

    adn = np.random.randn(batch_size, num_frame, num_freq) \
            + 1j*np.random.randn(batch_size, num_frame, num_freq)
    print('adn', adn.shape)
    norm_adn = np.linalg.norm(adn, ord='fro', axis = (1,2))
    print('norm_adn', norm_adn.shape)
    adn = adn / norm_adn[:, np.newaxis, np.newaxis] \
            * 10.**(-1.*(np.random.rand()*(SNR_MAX-SNR_MIN)+SNR_MIN)/20)
    print('adn', adn.shape)
    
    break