import numpy as np
from util import modified_SMAPE
from keras.callbacks import Callback

class EvaluateSMAPE(Callback):
    def __init__(self, xs, ys_true, name, scaler_y=None, use_SMAPE_loss=False):
        self.scaler_y = scaler_y
        self.xs = xs
        self.ys_true = np.squeeze(scaler_y.inverse_transform(ys_true))
        self.name = name
        self.use_SMAPE_loss = use_SMAPE_loss

    def on_epoch_end(self, epoch, logs=None):
        if self.use_SMAPE_loss:
            score = 100 - 100*logs[f'{self.name}_loss']
        else:
            ys_pred = self.model.predict(self.xs)
            if self.scaler_y is not None:
                ys_pred = np.squeeze(self.scaler_y.inverse_transform(ys_pred))
            score = modified_SMAPE(self.ys_true, ys_pred)
        logs[f'{self.name}_SMAPE'] = score
        print(f'Epoch {epoch+1} | {self.name}-SMAPE: {score}')
