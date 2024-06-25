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
from torchvision import transforms
import cv2
from sklearn.model_selection import StratifiedKFold

def scalRadius(img, scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    if r < 0.001: # This is for the very black images
        r = scale*2
    s = scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)


def load_img_fundus(dir_img):
    scale = 300
    a = np.load(dir_img)
    a = scalRadius(a,scale)
    a = cv2.addWeighted(a,4,cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1]/2),int(a.shape[0]/2)), int(scale*0.9), (1,1,1), -1, 8, 0)
    a = a*b + 128*(1-b)
    img = Image.fromarray(np.array(a, dtype=np.int8), "RGB")
    return img

def load_img_oct(img_path):
     oct_array = np.load(img_path)
     oct_img = torch.from_numpy(oct_array.astype('float32'))  # .item().volume).astype('float32'))
     oct_img_resized = F.resize(oct_img,(224,224))
     oct_img_resized = oct_img_resized[0:128] #0:10]   #nlayer]
     return oct_img_resized

class retina(data.Dataset):
    """ Multi-Modal Dataset.
        Args:
        dir_imgs (string): Root directory of dataset where images exist.
        transform_fundus: Tranformation applied to fundus images

        ids_set (pandas DataFrame):
    """

    def __init__(self,
                dir_imgs_fundus,
                dir_imgs_oct,
                ids_set
                ):
        self.path_imgs_fundus = []
        self.path_imgs_oct = []
        self.ids = []
        ids_set = ids_set.reset_index(drop=True)
        self.dir_imgs_fundus = dir_imgs_fundus
        self.dir_imgs_oct = dir_imgs_oct
        for idx, ID in enumerate(ids_set['ID'].values):

            # Reading all oct images per patient
            imgs_per_id_fundus = glob.glob(self.dir_imgs_fundus + '/*.npy')
            img_fundus = [j for j in imgs_per_id_fundus if str(int(ID)) in j]

            imgs_per_id_oct = glob.glob(self.dir_imgs_oct + '/*.npy')
            img_oct = [j for j in imgs_per_id_oct if str(int(ID)) in j]
            if img_fundus:
                imgs_per_id_ = str(img_fundus[0])
                self.path_imgs_fundus.append(imgs_per_id_)
        #        self.labels.append(diagnoses[idx])
                self.ids.append(imgs_per_id_.split('/')[-1].split('_')[1]) 
            else:
                continue
            if img_oct:
                imgs_per_id__ = str(img_oct[0])
                self.path_imgs_oct.append(imgs_per_id__)
            else:
                continue
        
        self.transform_fundus = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            
    # Denotes the total number of samples
    def __len__(self):
        return len(self.path_imgs_fundus) # self.num_parti

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (tuple): Index
        Returns:
            tuple: (oct, label)
        """
        fundus = load_img_fundus(self.path_imgs_fundus[index])
        fundus_left = self.transform_fundus(fundus)
        fundus_imag_left = (fundus_left - torch.min(fundus_left))/(torch.max(fundus_left) - torch.min(fundus_left))
        oct_left = load_img_oct(self.path_imgs_oct[index])
        oct_imag_left = (oct_left - torch.min(oct_left))/(torch.max(oct_left) - torch.min(oct_left))
        return fundus_imag_left, oct_imag_left, self.ids[index]
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


class retina_DM(pl.LightningDataModule):    
        
    def __init__(self, 
        data_dir_fundus: Union[None, str] = None,
        data_dir_oct: Union[None, str] = None,
        include_ids: Union[None, List[int], str] = None,
        exclude_ids: Union[None, List[int], str] = None,
        split_lengths: Union[None, List[int]]=None,
        shuffle=False,
        img_size_fundus: Union[None, List[int], str] = None,
        img_size_oct: Union[None, List[int], str] = None,
        batch_size: Union[None, int] = None
    ):

        '''
        params:
            data_dir:
            batch_size:
            split_lengths:
        '''
        
        super().__init__()
        self.data_dir_fundus = data_dir_fundus
        self.data_dir_oct = data_dir_oct
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
        imgs = retina(self.data_dir_fundus, self.data_dir_oct, self.include_ids)
        print('Found' + str(len(imgs)) + ' fundus images')
#        embed()

        
        if self.split_lengths is None:
            train_len = int(0.6 * len(imgs))
            test_len = int(0.1 * len(imgs))
            val_len = len(imgs) - train_len - test_len
            self.split_lengths = [train_len, val_len, test_len]

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(imgs, self.split_lengths)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=self.shuffle, collate_fn=self.my_collate, drop_last=True)
        
