import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class ohkdd_valid(Dataset):
    def __init__(self, data, tar, use_gpu=False):
        self.data = data
        self.tar = tar
        if torch.cuda.is_available() and use_gpu:
            self.data = self.data.cuda()
            self.tar = self.tar.cuda()

        self.data = F.normalize(self.data)

    def __getitem__(self, item):
        return self.data[item], self.tar[item]

    def __len__(self):
        return len(self.data)


class ohkdd(Dataset):
    def __init__(self, test_size=0.2, use_gpu=False, data_path="kdd99_oh.npy", tar_path="kdd99_oh_label.npy",
                 return_type=1):
        self.use_gpu = use_gpu
        self.test_size = test_size
        data = np.load(data_path)
        label = np.load(tar_path, allow_pickle=True)

        if return_type == 2:
            size = 1
            while size * size < len(data[0]):
                size += 1

            data = np.pad(data, ((0, 0), (0, size * size - len(data[0]))))
            data.resize((len(data), size, size))

        if test_size != 0:
            self.data_x, self.test_x, self.data_y, self.test_y = \
                train_test_split(data, label, test_size=test_size)
            self.data_x = torch.unsqueeze(torch.from_numpy(self.data_x), dim=1).float()
            self.test_x = torch.unsqueeze(torch.from_numpy(self.test_x), dim=1).float()
            self.data_y = torch.from_numpy(self.data_y.astype(int)).long()
            self.test_y = torch.from_numpy(self.test_y.astype(int)).long()
            self.test_x = F.normalize(self.test_x)
        else:
            self.data_x = torch.unsqueeze(torch.from_numpy(data), dim=1).float()
            self.data_y = torch.from_numpy(label.astype(int)).long().float()

        self.data_x = F.normalize(self.data_x)
        if torch.cuda.is_available() and use_gpu:
            self.data_x = self.data_x.cuda()
            self.data_y = self.data_y.cuda()
            if test_size != 0:
                self.test_x = self.test_x.cuda()
                self.test_y = self.test_y.cuda()

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return len(self.data_x)

    def get_valid(self):
        if self.test_size == 0:
            raise UserWarning("test_size should not be 0!!")
        return ohkdd_valid(self.test_x, self.test_y, use_gpu=self.use_gpu)


if __name__ == "__main__":
    mod = ohkdd(return_type=2)
    print(len(mod), len(mod.get_valid()))
    print(mod.data_x.shape)
    mod = ohkdd(test_size=0, return_type=2)
    print(len(mod))
    print(mod.data_x.shape)
