import os
import glob
from data import common
import numpy as np
import torch.utils.data as data
import torch
import math
import random
import pydicom
from pathlib import Path
import time

class SRData(data.Dataset):
    def __init__(self, config, mode='train', patient_ID=None, augment=False, cache='none'):
        self.data_spec = config.get('dataset')
        self.model_spec = config.get('model')
        self.mode = mode
        self.patient_ID = patient_ID
        self.augment = augment
        self.cache = cache
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')
        self.names_ldct, self.names_ndct = self._scan()

    def __getitem__(self, idx):
        ldct, ndct loc_name = self._load_file(idx)
        ldct, ndct = self.preparation(ldct, ndct)
        return ldct, ndct, loc_name

    def __len__(self):
        if self.mode == 'train':
            self.dataset_length = int(len(self.names_img) * self.data_spec['train']['repeat'])
        elif self.mode == 'valid':
            self.dataset_length = len(self.names_img)
        else:
            self.dataset_length = len(self.names_img)
        return self.dataset_length

    def _scan(self):
        names_ldct = []
        names_ndct = []
        for ID in self.patient_ID:
            data_path = os.path.join(self.data_spec['data_dir'], ID, ID)
            names_ldct += glob.glob(os.path.join(data_path, 'quarter_1mm', '*.IMA'))
            names_ndct += glob.glob(os.path.join(data_path, 'full_1mm', '*.IMA'))
        return names_ldct, names_ndct

    def _get_index(self, idx):
        return idx % len(self.names_img)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        loc_name = self.names_ndct[idx]

        ldct = np.load(self.names_ldct[idx]).astype(np.float32)
        ndct = np.load(self.names_ndct[idx]).astype(np.float32)

        loc_name = Path(loc_name).parts[-3::]
        return ldct, ndct, loc_name

    def preparation(self, ldct, ndct):
        if self.mode == 'train' and self.augment:
            ldct, ndct = common.augment(ldct, ndct)
        sino, img, grid_norm = common.get_patch(
            np.expand_dims(ldct, 0), np.expand_dims(ndct, 0)
        )
        return ldct, ndct
