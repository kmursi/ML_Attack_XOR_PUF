import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from models import TimeHistory
from models import myCallback



class XOR_PUF(BaseEstimator, ClassifierMixin):

    """
    *************************************************************************************
    *                                Model initialization                               *
    *************************************************************************************
    """
    def __init__(self, stages = 64, streams = 4, epochs=100, plot=0, fig_name='', patience = 5):
        self.stage = stages
        self.streams = streams
        self.epochs=epochs
        self.plot = plot
        self.fig_name = fig_name
        self.patience = patience

    """
    *************************************************************************************
    *                                    Model plotting                                 *
    *************************************************************************************
    """
    def plotting(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'], label="training accuracy", color='red', linewidth=2, linestyle=':')
        plt.plot(history.history['val_accuracy'], label="validation accuracy", color='blue', linewidth=2)
        plt.plot(history.history['loss'], label="training loss", color='#4b0082', linewidth=2,
                 linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
        plt.plot(history.history['val_loss'], label="validation loss", color='coral', linewidth=2)
        plt.title('%1.0f-XPUF 64-bit model accuracy'%self.streams)
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['Tr. Acc.', 'Val. Acc.', 'Tr. Loss', 'Val. Loss'], loc='upper left')
        plt.savefig(self.fig_name)




    """
    *************************************************************************************
    *                        Model compiling, fitting, and evaluating                   *
    *************************************************************************************
    """
    def fit(self, X, y):
        time_callback = TimeHistory.TimeHistory()
        # convert into 0 and 1
        y = (np.array(y) + 1) / 2.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

        '************************* setup target training accuracy **************************************************'
        callbacks = myCallback.myCallback(0.98, self.patience)

        self.model = tf.keras.Sequential()

        self.model.add(layers.Dense(int(2 ** (self.streams) / 2), activation='tanh', input_dim=(self.stage),
                                    kernel_initializer='random_normal'))
        self.model.add(layers.Dense(int(2 ** (self.streams)), activation='tanh'))
        self.model.add(layers.Dense(int(2 ** (self.streams)/2), activation='tanh'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        '************************* training **************************************************'
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, callbacks=[time_callback, callbacks],
                                 shuffle=True, validation_split=0.01)
        if self.plot == 1:
            self.plotting(history)

        '************************* evaluate NN **************************************************'
        # score = self.model.score(X_test, y_test)
        results = self.model.evaluate(X_test, y_test, batch_size=128)
        print('\n')
        print('********** Model results **********')
        print('Test Loss = %2.4f'%float(results[0]))
        print('Test Acc = %2.2f%% '%float(results[1]*100))
        print('Elapsed Time: %.3f sec'% np.sum(time_callback.times))
        print('The plot is saved in the main folder successfully!')

