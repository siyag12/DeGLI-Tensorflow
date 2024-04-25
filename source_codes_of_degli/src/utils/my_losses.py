# -*- coding: utf-8 -*-
##############################################################################
import tensorflow as tf


def phase_sensitive_l2(x_ora, x_est):
    return tf.reduce_mean(tf.square(tf.abs(x_ora-x_est)))