# Import libraries and classes
import os
import numpy as np
import torch

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

class Dataset:
    def __init__(self,datapath):
        self.dataloaders = []
        self.dataset_sizes = []
        self.load(datapath)

    # Function to read the dataset from the given specified directory and return dataloaders and dataset sizes. (Reference for some of the code: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
    def load(self,datapath):
        # reshape the img to the desired size, and do the normalization
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # this filter is normally used in rgb img, https://pytorch.org/vision/stable/models.html
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
        # ImageFolder expects data loader as:
        #     datapath/train/class1/xxx.png
        #     datapath/train/class2/xxx.png
        #     ...
        #     datapath/test/class1/xxx.png
        #     ...

        # Load train and test datasets using torchvision.datasets.ImageFolder class
        train_dataset = ImageFolder(root=os.path.join(datapath, 'train'), transform=data_transforms['train'])
        test_dataset = ImageFolder(root=os.path.join(datapath, 'test'), transform=data_transforms['test'])

        # Split a training set and a validation set from the original training set
        train_size = int(0.8 * len(train_dataset))
        validation_size = len(train_dataset) - train_size
        train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size])

        # Create data loaders for the train, validation, and test datasets with a batch size of 64
        batch_size = 64
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_subset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Store the dataloaders and the size of each dataset
        self.dataloaders = {'train': train_dataloader,
                            'validation': validation_dataloader,
                            'test': test_dataloader}
        self.dataset_sizes = {'train': len(train_subset),
                            'validation': len(validation_subset),
                            'test': len(test_dataset)}
        
    def describe(self):
        for dataset in self.dataset_sizes:
            print(f"{dataset} size: {self.dataset_sizes[dataset]}")
            for X, y in self.dataloaders[dataset]:
                print(f"Shape of X [N, C, H, W]: {X.shape}")
                print(f"Shape of y: {y.shape} {y.dtype}")
                break