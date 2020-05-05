from tensorflow import keras

"""
*************************************************************************************
*                        Stop training based on predefined val_acc                  *
*************************************************************************************
"""


class myCallback(keras.callbacks.Callback):

    def __init__(self, acc_threshold, patience):
        self.acc_threshold=acc_threshold
        self.patience = patience
        self.defaultPatience =patience
        self.previous_accuracy = 0.0

    def on_epoch_end(self, epoch, logs={}):
        try:
            '''
            Stop the training when the validation acc reached the predefined 98%
            '''
            if(float(logs.get('val_accuracy')) > float(self.acc_threshold)):
                print("\nReached %2.2f%% accuracy, so stopping training!!" %(self.acc_threshold*100))
                self.model.stop_training = True

            '''
            Stop the training when the validation acc is not enhancing for consecutive patience value
            '''
            if (int(logs.get('val_accuracy')) < int(self.previous_accuracy)):
                self.patience -= 1
                if (self.patience == 0):
                    print('\n*************************************************************************************')
                    print('************** Break the training because of early stopping! *************************')
                    print('*************************************************************************************\n')
                    self.model.stop_training = True
            else:
                '''
                Reset the patience value if the learning enhanced!
                '''
                self.patience = self.defaultPatience
            self.previous_accuracy = logs.get('accuracy')

        except:
            print("\n An exception occurred !! (myCallback) class !! \n")