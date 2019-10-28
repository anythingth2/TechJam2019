import keras.backend as K
import numpy as np

def SMAPE_loss(scaler):
    mean = K.constant(scaler.mean_)
    std = K.constant(np.sqrt(scaler.var_))
    def loss_func(y_true, y_pred):
        y_true = (y_true * std) + mean
        y_pred = (y_pred * std) + mean
        return K.mean((K.abs(y_pred - y_true) ** 2) / (( K.minimum(K.abs(y_true)*2, K.abs(y_pred)) + K.abs(y_true)) ** 2)) 
    return loss_func