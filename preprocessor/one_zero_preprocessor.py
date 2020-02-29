# -*- coding: utf-8 -*-

class OneZeroPreprocessor:
    '''
    Divide the value of each pixel to 255
    to scale it back to [0, 1] range.
    '''

    def process(self, images):
        return images / 255.0