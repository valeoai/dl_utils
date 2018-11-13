import os
import shutil

import keras
import numpy as np
from PIL import Image


def create_subdataset(path):
    os.makedirs(path / 'subdataset/train/dogs')
    os.makedirs(path / 'subdataset/train/cats')
    os.makedirs(path / 'subdataset/valid/dogs')
    os.makedirs(path / 'subdataset/valid/cats')

    input_path = path / 'train/cats'
    output_path = path / 'subdataset/train/cats'
    for f in os.listdir(input_path)[:2000]:
        shutil.copyfile(input_path / f, output_path / f)

    input_path = path / 'train/dogs'
    output_path = path / 'subdataset/train/dogs'
    for f in os.listdir(input_path)[:2000]:
        shutil.copyfile(input_path / f, output_path / f)

    input_path = path / 'valid/cats'
    output_path = path / 'subdataset/valid/cats'
    for f in os.listdir(input_path)[0:500]:
        shutil.copyfile(input_path / f, output_path / f)

    input_path = path / 'valid/dogs'
    output_path = path / 'subdataset/valid/dogs'
    for f in os.listdir(input_path)[0:500]:
        shutil.copyfile(input_path / f, output_path / f)


class GeneratorSingleObject(keras.utils.Sequence):
    """Generates data from a Dataframe"""

    def __init__(self, df, folder, preprocess_fct, batch_size=32, dim=(32, 32),
                 shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = folder
        self.class_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                           'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                           'person', 'pottedplant', 'sheep',
                           'sofa', 'train', 'tvmonitor']
        self.NbClasses = len(self.class_name)
        self.class_dict = dict(
            (self.class_name[o], o) for o in range(self.NbClasses))

        self.df = df
        self.n = len(df)
        self.nb_iteration = int(np.floor(self.n / self.batch_size))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.nb_iteration

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def next_item(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        """Generates data containing batch_size samples"""
        # Initialization
        # X: (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, 3))
        Y_bb = np.zeros((self.batch_size, 4))
        Y_clas = np.zeros((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = Image.open(self.folder / self.df['filename'][ID])
            bb = self.df['bbox'][ID]
            bb = np.fromstring(bb, dtype=np.int, sep=' ')

            width, height = img.size
            RatioX = width / self.dim[0]
            RatioY = height / self.dim[1]

            img = np.asarray(img.resize(self.dim))
            bb = [bb[0] / RatioY, bb[1] / RatioX, bb[2] / RatioY, bb[3] / RatioX]

            X[i] = self.preprocess_fct(np.asarray(img))
            Y_bb[i] = bb
            Y_clas[i] = self.class_dict[self.df['cat'][ID]]

        Y_clas = keras.utils.to_categorical(Y_clas, self.NbClasses)

        return X, [Y_bb, Y_clas]
