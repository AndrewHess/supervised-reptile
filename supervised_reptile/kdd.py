"""
Loading and augmenting the KDD dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import csv
import os
import random
import numpy as np

def read_dataset(data_dir):
    """
    Iterate over the data in a data directory.

    Args:
      data_dir: a directory of activity type directories.

    Returns:
      An iterable over activity data.
    """
    for activity_type in sorted(os.listdir(data_dir)):
        activity_dir = os.path.join(data_dir, activity_type)
        if not os.path.isdir(activity_dir):
            continue
        yield KDDData(activity_dir)

        '''
        for data_file in sorted(os.listdir(activity_dir)):
            if not data_file.startswith('data'):
                continue
            yield KDDData(os.path.join(activity_dir, data_file))
        '''

def split_dataset(dataset, num_train=10):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of KDDDatas.

    Returns:
      A tuple (train, test) of KDDData sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

# pylint: disable=R0903
class KDDData:
    """
    A single kdd datapoint class.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_points):
        """
        Sample data points (as numpy arrays) from the class.

        Returns:
          A sequence of 4x4 numpy arrays.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.csv')]
        random.shuffle(names)
        data = []
        for name in names[:num_points]:
            data.append(self._read_data(os.path.join(self.dir_path, name)))
        return data

    def _read_data(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'r') as in_file:
            # Read the csv file as a numpy array.
            reader = csv.reader(in_file, delimiter=',')
            counter = 0
            for row in reader:
                self._cache[path] = np.array(row).astype('float32')
                self._cache[path].resize((16, 1))

        return self._cache[path]
