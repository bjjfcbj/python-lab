import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class Kdd99TrainSet(Dataset):
    def __init__(self):
        self.raw_train_data = pd.read_csv('C:\\Users\Administrator\Downloads\data\kdd99\kdddata_nodup.csv')
        self.raw_test_data = pd.read_csv('C:\\Users\Administrator\Downloads\data\kdd99\corrected', header=None)

        self.obj_col = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
        self.dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
        self.u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
        self.r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
        self.probe = ['ipsweep', 'nmap', 'portsweep', 'satan']

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    pass
