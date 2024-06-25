import numpy as np
import sys
import torch.utils.data as data
import torch
import glob
import pandas as pd
import pdb
import sys
from itertools import chain, combinations
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from IPython import embed
from PIL import Image
from sklearn.model_selection import StratifiedKFold

def load_img(img_path):
     oct_array = np.load(img_path)
     oct_img = torch.from_numpy(oct_array.astype('float32'))  # .item().volume).astype('float32'))
     oct_img_resized = F.resize(oct_img,(224,224))
     oct_img_resized = oct_img_resized[0:128] #0:10]   #nlayer]
     return oct_img_resized


class OCT(data.Dataset):
    """ Multi-Modal Dataset.
        Args:
        dir_imgs (string): Root directory of dataset where images exist.
        transform_oct: Tranformation applied to oct images

        ids_set (pandas DataFrame):
    """

    def __init__(self,
                dir_imgs_left,
                ids_set
                ):

        self.labels = []        
        self.path_imgs_oct_left = []
        self.ids = []
        ids_set = ids_set.reset_index(drop=True)
        for idx, ID in enumerate(ids_set['ID'].values):

            # Reading all oct images per patient
            imgs_per_id_left = glob.glob(dir_imgs_left + '/*.npy')
            img_oct_left = [j for j in imgs_per_id_left if str(int(ID)) in j]
            if img_oct_left:
                imgs_per_id_left = str(img_oct_left[0])
                self.path_imgs_oct_left.append(imgs_per_id_left)
        #        self.labels.append(diagnoses[idx])
                self.ids.append(imgs_per_id_left.split('/')[-1].split('_')[1]) 
            else:
                continue
    # Denotes the total number of samples
    def __len__(self):
        return len(self.path_imgs_oct_left) # self.num_parti

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (tuple): Index
        Returns:
            tuple: (oct, label)
        """
        oct_left = load_img(self.path_imgs_oct_left[index])
        oct_imag_left = (oct_left - torch.min(oct_left))/(torch.max(oct_left) - torch.min(oct_left)) # Normalize between 0 and 1
        return oct_imag_left, self.ids[index]
#        else:
#           return None

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.n_batches)  #y)


class OCT_DM(pl.LightningDataModule):    
        
    def __init__(self, 
        data_dir_left: Union[None, str] = None,
        include_ids: Union[None, List[int], str] = None,
        exclude_ids: Union[None, List[int], str] = None,
        split_lengths: Union[None, List[int]]=None,
        shuffle=False,
        img_size: Union[None, List[int], str] = None,
        batch_size: Union[None, int] = None
    ):

        '''
        params:
            data_dir:
            batch_size:
            split_lengths:
        '''
        
        super().__init__()
        self.data_dir_left = data_dir_left
        self.include_ids = include_ids
        ## generate list of individuals to include
        if include_ids is not None and isinstance(include_ids, list):
             self.include_ids = include_ids
        if include_ids is not None and isinstance(include_ids, str):
             self.include_ids = pd.read_csv(include_ids) #, usecols='IDs') 
        else:
             self.exclude_ids = exclude_ids
        #
        ## the same for exclude_ids
        #

        self.batch_size = batch_size
        self.split_lengths = split_lengths
        self.shuffle = shuffle
        self.img_size = img_size


    def my_collate(self, batch):
       len_batch = len(batch) 
       batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
       if len_batch > len(batch): # source all the required samples from the original dataset at random
         diff = len_batch - len(batch)
         for i in range(diff):
           batch = batch + batch[:diff]
       return torch.utils.data.dataloader.default_collate(batch)

    def setup(self, stage: Optional[str] = None):
        
        #
        # read images
        #
        # TODO: add logic for exclude_ids
        imgs = OCT(self.data_dir_left, self.include_ids)
        print('Found' + str(len(imgs)) + ' oct images')
#        embed()

        
        if self.split_lengths is None:
            train_len = int(0.5 * len(imgs))
#            test_len = int(1 * len(imgs))
            val_len = len(imgs) - train_len #- test_len
            self.split_lengths = [train_len, val_len]

        self.train_dataset, self.val_dataset = random_split(imgs, self.split_lengths)
        self.test_dataset = imgs
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)
        
