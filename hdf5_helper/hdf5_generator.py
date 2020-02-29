# -*- coding: utf-8 -*-

import h5py
import numpy as np
from keras.utils import to_categorical


class HDF5Generator:
    '''
    Generator to generate training/validation/test data from hdf5 file.
    '''

    def __init__(self, db_path, is_categorical, batch_size=128, preprocessors=None, augumentator=None):
        self.batch_size = batch_size
        self.is_categorical = is_categorical
        self.preprocessors = preprocessors
        self.augumentator = augumentator

        self.db = h5py.File(db_path)
        self.db_size = self.db['labels'].shape[0]

    def generator(self, max_epochs=np.inf):
        '''
        Create generator object to load and process data from hdf5 file.
        '''

        epochs = 0

        # If max_epochs is set, only generate data for maximum "max_epochs" times.
        while epochs < max_epochs:
            for i in np.arange(0, self.db_size, self.batch_size):
                images = self.db['data'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]

                # Use one-hot encoding on labels if needed
                if self.is_categorical:
                    labels = to_categorical(labels)

                # Pass data through all preprocessor object to normalize data
                if self.preprocessors is not None:
                    for preprocessor in self.preprocessors:
                        images = preprocessor.process(images)

                # Perform data augumentation if needed
                if self.augumentator is not None:
                    images, labels = next(self.augumentator.flow(images, labels, batch_size=self.batch_size))
                
                yield images, labels
            
            epochs += 1
    
    def close(self):
        self.db.close()
