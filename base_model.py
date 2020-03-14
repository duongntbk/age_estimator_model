# -*- coding: utf-8 -*-

from abc import abstractmethod
from os.path import join as path_join

from keras import backend
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from callback.epoch_checkpoint import EpochCheckpoint
from callback.training_monitor import TrainingMonitor
from hdf5_helper.hdf5_generator import HDF5Generator
from preprocessor.mean_preprocessor import MeanPreprocessor
from preprocessor.divide_preprocessor import DividePreprocessor


class BaseModel:
    '''
    Base class for both gender predicting and age guessing.
    '''

    def __init__(self, conv_base, num_gpus, train_generator=None, validation_generator=None, test_generator=None):
        '''
        If train_generator is not specified,
        a generator which apply augmentation and both preprocessors will be created.
        Validation_generator and test_generator can also be created automatically,
        but we do not apply augmentation on validation_generator and test_generator.
        '''

        backend.set_learning_phase(0) # must set this to enable training on ResNet
        self.conv_base = conv_base # the pre-trained network to be tuned
        self.num_gpus = num_gpus # mxnet backend can support parallel tranining on multi gpus

        mean_json_path = path_join(self.hdf5_path, 'mean.json')

        if train_generator is None:
            aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                horizontal_flip=True, fill_mode="nearest")

            training_path = path_join(self.hdf5_path, 'training.hdf5')
            self.train_generator = HDF5Generator(training_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)],
                augmentator=aug)
        else:
            self.train_generator = train_generator

        if validation_generator is None:
            validation_path = path_join(self.hdf5_path, 'validation.hdf5')
            self.validation_generator = HDF5Generator(validation_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)])
        else:
            self.validation_generator = validation_generator
        
        if test_generator is None:
            test_path = path_join(self.hdf5_path, 'test.hdf5')
            self.test_generator = HDF5Generator(test_path, batch_size=64,
                is_categorical=False,
                preprocessors=[MeanPreprocessor(mean_json_path), DividePreprocessor(127.5)])
        else:
            self.test_generator = test_generator

    def build_tuning_model(self, head_path, trainable_layers, learning_rate=1e-3):
        '''
        Build model to be tuned.
        We un-freeze the few top layers of pre-trained network and tune those layers.
        Learning rate can also be specified here.
        '''
    
        model = load_model(head_path)
        base_depth = len(self.conv_base.layers)

        for layer in model.layers[0:base_depth-trainable_layers]:
            layer.trainable = False
        for layer in model.layers[base_depth-trainable_layers:base_depth]:
            layer.trainable = True

        model = self.compile_tuning(model, learning_rate)

        return model

    def fit(self, model, name, epochs=25, start_epochs=0, class_weight=None):
        '''
        Start training on a model.
        We log training loss/accuracy and validation loss/accuracy to disk
        to monitor the training process.
        If we are resuming training, we can specify the current epochs using "start_epochs".
        '''
    
        graph_path = path_join(self.model_path, '{0}.png'.format(name))
        json_path = path_join(self.model_path, '{0}.json'.format(name))
        training_monitor = TrainingMonitor(output_path=graph_path, json_path=json_path,
            start_epoch=start_epochs, metric_name=self.metric_name)

        epoch_checkpoint = EpochCheckpoint(self.model_path, every=2, start_at=start_epochs)

        model.fit_generator(
            self.train_generator.generator(),
            steps_per_epoch=self.train_generator.db_size // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator.generator(),
            validation_steps=self.validation_generator.db_size // self.validation_generator.batch_size,
            class_weight=class_weight,
            callbacks=[training_monitor, epoch_checkpoint]
        )

    def test(self, model_path):
        '''
        Load a trained model from path and test it on test data set.
        '''
    
        model = load_model(model_path)

        return model.evaluate_generator(
            self.test_generator.generator(), 
            steps=self.test_generator.db_size // self.test_generator.batch_size, 
        )

    @abstractmethod
    def build_warmup_model(self, learning_rate=1e-3):
        '''
        This method stack a few fully connected layers on top of a pre-trained network.
        This time we keep all layers in pre-trained network frozen
        and only train those fully connected layers.
        This is a preparation before full-blown fine-tuning.
        '''

        return None

    @abstractmethod
    def compile_tuning(self, model, learning_rate=1e-3):
        '''
        Compile fine tuning network with necessary loss function, metric,...
        '''

        return None
