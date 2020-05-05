
from tensorflow import keras
import time

class TimeHistory(keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super(TimeHistory, self).__init__(*args, **kwargs)

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)