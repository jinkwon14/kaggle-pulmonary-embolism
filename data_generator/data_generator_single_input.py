import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, labels, data_path, batch_size=32, dim=(512,512), n_channels=3, n_classes = 12, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.data_path = data_path
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, [y_rspe, y_lspe, y_cpe, y_acpe, y_cnpe, y_qamo, y_qaco, y_rv_gte1, y_rv_lt1] = self.__data_generation(list_IDs_temp)

        return X, [y_rspe, y_lspe, y_cpe, y_acpe, y_cnpe, y_qamo, y_qaco, y_rv_gte1, y_rv_lt1]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y_rspe = np.empty((self.batch_size, 1), dtype=int)
        y_lspe = np.empty((self.batch_size, 1), dtype=int)
        y_cpe = np.empty((self.batch_size, 1), dtype=int)
        y_acpe = np.empty((self.batch_size, 1), dtype=int)
        y_cnpe = np.empty((self.batch_size, 1), dtype=int)
        y_qamo = np.empty((self.batch_size, 1), dtype=int)
        y_qaco = np.empty((self.batch_size, 1), dtype=int)
        y_rv_gte1 = np.empty((self.batch_size, 1), dtype=int)
        y_rv_lt1 = np.empty((self.batch_size, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             print(self.data_path + ID + '.png')
            img = np.load(self.data_path + ID + '.npy',allow_pickle=True)
            # img = np.array(img)
            img = img/255
            X[i,] = img

            # img = np.load(self.data_path + ID + '.png')
#             print(type(img))


            # Store class
            y_rspe[i], y_lspe[i], y_cpe[i], y_acpe[i], y_cnpe[i], y_qamo[i], y_qaco[i], y_rv_gte1[i], y_rv_lt1[i] = self.labels[ID]

        return X, [y_rspe, y_lspe, y_cpe, y_acpe, y_cnpe, y_qamo, y_qaco, y_rv_gte1, y_rv_lt1]
