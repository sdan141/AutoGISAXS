# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy
import sqlite3
import pandas
import fabio
import glob

class Experiment:
    def __init__(self, data_path, fast_real):
        self.data_path = data_path['path']
        self.fast = fast_real

    def get_images(self, sum_every=False):
        if self.fast:
            self.load_real()
        images = []
        files = glob.glob(self.data_path + '*.cbf')
        files = sorted(glob.glob(self.data_path + '*.cbf'))

        for file in files:
            # flip top and bottom (vertical flip in y-direction)
            image = numpy.flip(fabio.open(file).data, axis=0)
            image.astype(numpy.float32)
            images.append(image.astype(numpy.float32))
        if sum_every:
            images = [sum(images[i:i+10]) for i in range(0,len(images),10)]

        return images, files

    def load_real(self):
        image_file = glob.glob(self.data_path + '*.npy')
        files = glob.glob(self.data_path + '*.csv')
        with open(image_file) as f:
            images = numpy.load(f)
        if files:
            df = pandas.csv_read(files[0])
            files = df['file_name'].to_numpy()
        else:
            files = numpy.arange(len(images))

        return images, files
