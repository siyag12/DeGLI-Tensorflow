# -*- coding: utf-8 -*-
###############################################################################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
 

## Models #####################################################################   
class ComplexGatedConvAmpM():
    
    
    def __init__(self, num_ch, kernel_size=(3,5), strides=(1,1), scope=None):
        
        with tf.variable_scope(scope):
            
            self.conv1r = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv1r')
            self.conv1i = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv1i')
            self.conv1g = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 name=scope+'conv1g')
    
            self.conv2r = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv2r')
            self.conv2i = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv2i')
            self.conv2g = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 name=scope+'conv2g')
            
            self.conv3r = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv3r')
            self.conv3i = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 use_bias=False, name=scope+'conv3i')
            self.conv3g = Conv2D(num_ch, kernel_size, strides, padding='same',
                                 name=scope+'conv3g')

            
            self.conv4r  = Conv2D(1, (1,1), strides, padding='same',
                                  use_bias=False, name=scope+'conv4r')
            self.conv4i  = Conv2D(1, (1,1), strides, padding='same',
                                  use_bias=False, name=scope+'conv4i')
                        
    def __call__(self, x, amp):
        
        x_real = x[:,:,:,0:5:2]
        x_imag = x[:,:,:,1:6:2]
        ampg = tf.abs(amp)[:,:,:,None]
        
        h1r = self.conv1r(x_real) - self.conv1i(x_imag)
        h1i = self.conv1r(x_imag) + self.conv1i(x_real)
        h1g = self.conv1g(tf.concat([tf.abs(tf.complex(x_real, x_imag)), ampg],
                                    axis=3))
        f1r = h1r*tf.nn.sigmoid(h1g)
        f1i = h1i*tf.nn.sigmoid(h1g)
        
        h2r = self.conv2r(f1r) - self.conv2i(f1i)
        h2i = self.conv2r(f1i) + self.conv2i(f1r)
        h2g = self.conv2g(tf.concat([tf.abs(tf.complex(f1r, f1i)), ampg],
                                    axis=3))
        f2r = h2r*tf.nn.sigmoid(h2g)
        f2i = h2i*tf.nn.sigmoid(h2g)
        
        h3r = self.conv3r(f2r) - self.conv3i(f2i)
        h3i = self.conv3r(f2i) + self.conv3i(f2r)
        h3g = self.conv3g(tf.concat([tf.abs(tf.complex(f2r, f2i)), ampg],
                                    axis=3))
        f3r = h3r*tf.nn.sigmoid(h3g)
        f3i = h3i*tf.nn.sigmoid(h3g)
        
        h4r = self.conv4r(f3r) - self.conv4i(f3i)
        h4i = self.conv4r(f3i) + self.conv4i(f3r)

        y  = tf.cast(h4r[:,:,:,0], dtype=tf.complex64) + \
                1j*tf.cast(h4i[:,:,:,0], dtype=tf.complex64)
        
        return y