import numpy as np
from PIL import Image
from collections import defaultdict




class DataSet(object):
    def __init__(self, images, labels=None, fake_data=False, first_shuffle=False):
        # if fake_data:
        #     self._num_examples = 10000
        # else:
        #     if labels is not None:
        #         assert images.shape[0] == labels.shape[0], (
        #             "images.shape: %s labels.shape: %s" % (images.shape,
        #                                                    labels.shape))
        if(isinstance(images,type([]))):
            images = np.asarray(images)
        if(isinstance(labels,type([]))):
            labels = np.asarray(labels)

        self._num_examples = images.shape[0]
        # images = images.astype(np.float32)
        # images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if(first_shuffle):
            self._index_in_epoch = self._num_examples + 1




    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # print(self._index_in_epoch,self._epochs_completed,self._num_examples)
        return self._images[start:end], self._labels[start:end]



