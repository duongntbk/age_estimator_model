# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    '''
    Log trainning loss/accuracy and validation loss/accuracy to json and png file.
    If training metric is not accuracy (for example: mae), *metric_name* must be specified.
    If loss and accuracy's scale is too different, you can specified loss_func/metric_func
    to scale up/down as needed.
    '''

    def __init__(self, output_path, json_path, start_epoch=0, metric_name='acc',
            loss_func=None, metric_func=None):
        super(TrainingMonitor, self).__init__()
        self.output_path = output_path
        self.json_path = json_path
        self.start_epoch = start_epoch
        self.metric_name = metric_name
        self.loss_func = loss_func
        self.metric_func = metric_func

    def on_train_begin(self, logs=None):
        '''
        Initialize necessary properties when training starts.
        '''

        self.history = {}

        # If json file already existed, trim history up to *start_epoch*
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    self.history = json.loads(f.read())
                
                # Trim history to start_epoch
                if self.start_epoch > 0:
                    for key in self.history.keys():
                        self.history[key] = self.history[key][:self.start_epoch]

    def on_epoch_end(self, epoch, logs={}):
        '''
        At the end of each epoch, append current training loss/accuracy and
        validation loss/accuracy to json file.
        Also update png graph.
        '''

        for (key, val) in logs.items():
            hit = self.history.get(key, []) # return empty array if this key does not exist

            if type(val) is np.float32:
                val = val.astype('float64')

            hit.append(val)
            self.history[key] = hit

        if self.json_path is not None:
            with open(self.json_path, 'w') as f:
                f.write(json.dumps(self.history))

        if len(self.history['loss']) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.history['loss']))
            plt.style.use('ggplot')
            plt.figure()

            # Apply function on loss and/or accuracy if needed.
            drw_train_loss = self.loss_func(self.history['loss']) \
                if self.loss_func is not None \
                else self.history['loss']
            drw_val_loss = self.loss_func(self.history['val_loss']) \
                if self.loss_func is not None\
                else self.history['val_loss']
            drw_train_metric = self.metric_func(self.history[self.metric_name]) \
                if self.metric_func is not None \
                else self.history[self.metric_name]
            drw_val_metric = self.metric_func(self.history['val_{0}'.format(self.metric_name)]) \
                if self.metric_func is not None \
                else self.history['val_{0}'.format(self.metric_name)]

            plt.plot(N, drw_train_loss, label='train_loss')
            plt.plot(N, drw_val_loss, label='val_loss')
            plt.plot(N, drw_train_metric, label='train_{0}'.format(self.metric_name))
            plt.plot(N, drw_val_metric, label='val_{0}'.format(self.metric_name))
            plt.title('Training Loss and Accuracy [Epoch {}]'.format(
            len(self.history['loss'])))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()

            # save the figure
            plt.savefig(self.output_path)
            plt.close()
