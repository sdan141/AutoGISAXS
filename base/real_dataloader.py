# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy
import sqlite3
import pandas
import fabio
import glob

class Experiment:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_images(self):
        images = []
        files = glob.glob(self.data_path + 'data/unlabeled/' + '*.cbf')
        for file in files:
            # flip top and bottom (vertical flip in y-direction)
            image = numpy.flip(fabio.open(file).data, axis=0)
            image.astype(numpy.float32)
            images.append(image.astype(numpy.float32))
        return images, files
