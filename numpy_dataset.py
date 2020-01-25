import numpy as np
from torch.utils.data.dataset import Dataset


class NumpyDataset(Dataset):
    def __init__(self, numpy_array: np.array):
        self.np_array = numpy_array

    def __getitem__(self, index):
        return self.np_array[index, :]

    def __len__(self):
        return self.np_array.shape[0]
