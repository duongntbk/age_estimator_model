# -*- coding: utf-8 -*-

import os

import h5py


class HDF5Writer:
    '''
    Helper class to convert data into hdf5 format.
    '''

    def __init__(self, dims, output_dir, output_name, buffer_size=1000):
        '''
        Set output path, dimensions and buffer size for writer.
        Output folder will be created if needed.
        '''
        
        # Check and create output folder if it does not exist.
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        # Create database object with 2 datasets.
        # We will normalize out images data, because of that we use dtype==float32
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset('data', dims, dtype='float32')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')

        self.buffer_size = buffer_size
        self.buffer = {
            'data': [],
            'labels': [], 
        }
        self.idx = 0 # Index in database

    def flush(self):
        '''
        Write data from buffer to disk and reset buffer.
        '''

        # Write buffer to disk
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i

        # Reset buffer
        self.buffer = {
            'data': [],
            'labels': [], 
        }

    def write(self, rows, labels):
        '''
        Write multiple rows to buffer,
        then flush buffer to disk if fulls.
        '''

        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # Buffer is full, flush data to disk
        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    def write_labels_name(self, label_names):
        '''
        Write label names as text into a different dataset.
        '''

        names_type = h5py.special_dtype(vlen=str)
        names_length = len(label_names)
        names = self.db.create_dataset('label_names', names_length, dtype=names_type)
        names[:] = label_names

    def close(self):
        '''
        Finish writing to hdf5 db.
        '''

        # If buffer still contains data, flush it all to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # Close database
        self.db.close()
