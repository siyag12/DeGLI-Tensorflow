# -*- coding: utf-8 -*-
##############################################################################
import tensorflow as tf


## STFT and iSTFT ############################################################
class STFT():
    
    
    def __init__(self, winlen, shift, nfft):
        self.winlen = winlen
        self.shift = shift
        self.nfft = nfft
        
    def __call__(self, x):
        X = tf.contrib.signal.stft(x, self.winlen, self.shift, fft_length=self.nfft)
        return X


class iSTFT():
    
    
    def __init__(self, winlen, shift, nfft):
        self.winlen = winlen
        self.shift = shift
        self.nfft = nfft
        
    def __call__(self, X):
        win=tf.contrib.signal.inverse_stft_window_fn(self.shift)
        x = tf.contrib.signal.inverse_stft(X, self.winlen, self.shift, 
                                           fft_length=self.nfft, window_fn=win)
        return x
    
    
## DeGLI ######################################################################
class DeGLIBlock():
    
    
    def __init__(self, amp_ora, stft, istft, model):
        self.amp_ora = amp_ora
        self.stft = stft
        self.istft = istft
        self.model = model
        
    def __call__(self,x):
        x_pc2 = self.amp_ora * tf.sign(x)
        x_pc1 = self.stft(self.istft(x_pc2))
        
        feat_real0 = tf.expand_dims(tf.real(x),-1)
        feat_imag0 = tf.expand_dims(tf.imag(x),-1)
        feat_real1 = tf.expand_dims(tf.real(x_pc1),-1)
        feat_imag1 = tf.expand_dims(tf.imag(x_pc1),-1)
        feat_real2 = tf.expand_dims(tf.real(x_pc2),-1)
        feat_imag2 = tf.expand_dims(tf.imag(x_pc2),-1)
        feat_conc  = tf.concat([feat_real0, feat_imag0, feat_real1, feat_imag1,
                          feat_real2, feat_imag2], axis=3)
    
        x_dnn = self.model(feat_conc, amp=self.amp_ora)
        x_out = x_pc1 - x_dnn
        
        self.residual = x_dnn
        self.pc1 = x_pc1
        self.pc2 = x_pc2
        
        return x_out
    
    
class DeGLIBlock10():
    
    
    def __init__(self, amp_ora, stft, istft, model):
        self.amp_ora = amp_ora
        self.stft = stft
        self.istft = istft
        self.model = model
        
    def degli_block(self, x):
        x_pc2 = self.amp_ora * tf.sign(x)
        x_pc1 = self.stft(self.istft(x_pc2))
        
        feat_real0 = tf.expand_dims(tf.real(x),-1)
        feat_imag0 = tf.expand_dims(tf.imag(x),-1)
        feat_real1 = tf.expand_dims(tf.real(x_pc1),-1)
        feat_imag1 = tf.expand_dims(tf.imag(x_pc1),-1)
        feat_real2 = tf.expand_dims(tf.real(x_pc2),-1)
        feat_imag2 = tf.expand_dims(tf.imag(x_pc2),-1)
        feat_conc  = tf.concat([feat_real0, feat_imag0, feat_real1, feat_imag1,
                          feat_real2, feat_imag2], axis=3)
    
        x_dnn = self.model(feat_conc, amp=self.amp_ora)
        x_out = x_pc1 - x_dnn
        return x_out
                
    def __call__(self, x):
        x0 = self.degli_block(x)
        x1 = self.degli_block(x0)
        x2 = self.degli_block(x1)
        x3 = self.degli_block(x2)
        x4 = self.degli_block(x3)
        x5 = self.degli_block(x4)
        x6 = self.degli_block(x5)
        x7 = self.degli_block(x6)
        x8 = self.degli_block(x7)
        x9 = self.degli_block(x8)
        return x9
    
    
