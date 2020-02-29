# -*- coding: utf-8 -*-

import copy

from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.utils import multi_gpu_model

from base_model import BaseModel


class AgeModel(BaseModel):
    '''
    This class is used to train age guessing model.
    '''

    def __init__(self, conv_base, num_gpus=2, train_generator=None, validation_generator=None, test_generator=None):
        self.model_path = 'model/age'
        self.history_path = 'history/age'
        self.hdf5_path = 'hdf5/age'
        self.metric_name = 'mae'

        super().__init__(conv_base=conv_base, num_gpus=num_gpus, train_generator=train_generator,
            validation_generator=validation_generator, test_generator=test_generator)

    def build_warmup_model(self, learning_rate=1e-3):
        '''
        Implement abstract method of parent class.
        
        This method stack a few fully connected layers on top of a pre-trained network.
        This time we keep all layers in pre-trained network frozen
        and only train those fully connected layers.
        This is a preparation before full-blown fine-tuning.
        '''

        conv_base = copy.deepcopy(self.conv_base)
        for layer in conv_base.layers:
            # Freeze all layers in VGG19 network,
            # we are not performing fine-tuning yet
            layer.trainable = False

        input = conv_base.output
        input = Dropout(0.4)(input)
        input = Flatten()(input)
        input = Dense(256, activation='relu')(input)
        input = Dense(256, activation='relu')(input)
        output = Dense(1)(input) # This is a regression model, activation is not needed here

        model = Model(conv_base.input, output, name='gender_head')

        if self.num_gpus > 0:
            model = multi_gpu_model(model, self.num_gpus) # mxnet might use this?

        # Use Mean Squared Error as loss function and Mean Absolute Error as metric
        model.compile(loss='mse',
            optimizer=optimizers.RMSprop(lr=learning_rate),
            metrics=['mae']
        )

        return model

    def compile_tuning(self, model, learning_rate=1e-3):
        '''
        Implement abstract method of parent class.
        
        Compile fine tuning network with necessary loss function, metric,...
        '''

        if self.num_gpus > 0:
            model = multi_gpu_model(model, self.num_gpus) # mxnet might use this?

        model.compile(loss='mse',
            optimizer=optimizers.Adam(lr=learning_rate),
            metrics=['mae']
        )

        return model
