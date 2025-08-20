import pytorch_lightning as L
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
import random
import numpy as np
import os


class BRISCDataset(Dataset):
    """
    """

    def __init__(self, file_path: str, transform = None):
        """
        """        
        #
        self.dataset = datasets.ImageFolder(file_path, transform = transform)

    def __len__(self):
        """This function return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """This function returns the item at the given index.
        Arguments:
            idx   The data index.
        """
        return self.dataset[idx]
    
class BRISCDataModule(L.LightningDataModule):
    """This class is a version of the LightningDataModule, after initialization
    it allows to set the dataset and then train dataloader.
    Attributes:
        data_dir      The directory where we have the data.
        batch_size    The dimension of the file.
        preload_gpu   This is a boolean variable that tells whether we should tun on GPU.
        nw            This is the number of workers that load the dataset in parallel.
        dataset       This variable is the dataset itself (can be of 2 types).

    """

    def __init__( self, data_dir: str, batch_size: int = 32, num_workers: int = 0, val_split: float = 0.2,
                  image_size : int = 224 # Important that all images have the same dimension
                  #preload_gpu: bool = False,
                ):
        
        """This constructor initialize the main class attributes."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        #self.preload_gpu = preload_gpu
        self.nw = num_workers
        self.val_split = val_split
        self.image_size = image_size

        # Basic transformation.
        self.transform = v2.Compose([
            v2.Resize((512, 512), antialias=True),                    
            v2.RandomHorizontalFlip(p=0.5),                           
            v2.RandomRotation(degrees=10),                             
            v2.RandomAffine(degrees=0, scale=(0.9, 1.1)),              
            v2.ToDtype(torch.float32, scale=True),                    
            v2.Normalize(mean=[0.485, 0.456, 0.406],                  
                         std=[0.229, 0.224, 0.225]),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
    ])

    def setup(self, stage: str):
        """
        This function set the dataset.
        """
        train_path = os.path.join(self.data_dir, "train")
        full_train = datasets.ImageFolder(train_path, transform=self.transform)
        
        # Split train/val.
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size

        # Load val_dataset, train_dataset
        self.train_dataset, self.val_dataset = random_split(full_train, [train_size, val_size])

        # Load test_dataset
        test_path = os.path.join(self.data_dir, "test")
        self.test_dataset = datasets.ImageFolder(test_path, transform=self.transform)

    # Function which return the Dataloader of the training set
    def train_dataloader(self):
        """This function is used to train the class instance on the chosen dataset."""
        return DataLoader(self.train_dataset, num_workers=self.nw, batch_size=self.batch_size,
                                shuffle=True,pin_memory=True,               
                                )
    # Function which return the Dataloader of the validation set 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                           shuffle=False, num_workers=self.nw, pin_memory=True
                           )
    
    # Function which return the Dataloader of the test set
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                           shuffle=False, num_workers=self.nw, pin_memory=True
                           ) 

