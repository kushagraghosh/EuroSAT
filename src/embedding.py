import yaml
import os
import torch
import data
import model
from torchvision import models
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pickle

class EmbeddingSpace:

    def __init__(self):

        self.Model = model.Model()
        self.Model.load_state_dict(torch.load('trained_resnet_model.pth'))

        self.Model.model.fc = nn.Identity()
        # print(self.Model)

        # Set the model to evaluation mode
        self.Model.eval()

    def all_embeddings(self, datapath):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_dataset = ImageFolder(root=os.path.join(datapath, 'train'), transform=data_transforms['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        self.embeddings_dict = {}
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_dataloader, 0):
                images = images.to(device)
                outputs = self.Model(images)
                sample_fname, _ = train_dataloader.dataset.samples[i]
                self.embeddings_dict[sample_fname] = outputs
        print(self.embeddings_dict)

        with open('embeddings_dict.pkl', 'wb') as file:
            pickle.dump(self.embeddings_dict, file)
    
    def linear_interpolation(self):
        with open('embeddings_dict.pkl', 'rb') as file:
            self.embeddings_dict = pickle.load(file)

        image1 = '../data/Train_Test_Splits/train/Industrial/Industrial_12.jpg'
        image2 = '../data/Train_Test_Splits/train/Forest/Forest_1.jpg'

        tensor1 = self.embeddings_dict[image1]
        tensor2 = self.embeddings_dict[image2]

        for alpha in [0, 0.2, 0.4, 0.8, 1]:
            # Linear interpolation
            interpolated_tensor = torch.lerp(tensor1, tensor2, alpha)

            # Compute Euclidean distances
            distances = {key: torch.norm(interpolated_tensor - tensor) for key, tensor in self.embeddings_dict.items()}

            # Get keys of the top 5 closest points
            top5_closest_keys = sorted(distances, key=distances.get)[:5]

            # Print out the file names of the top 5 closest tensors for the current alpha
            print(f"\nAlpha = {alpha}")
            print("Interpolated Tensor:", interpolated_tensor)
            print("Top 5 Closest File Names:")
            for key in top5_closest_keys:
                print(key)


# This is needed to run this file as a script rather than import it as a module
if __name__ == "__main__":

    # Load the configuration file
    with open('../config.yaml') as p:
        config = yaml.safe_load(p)

    embeddings = EmbeddingSpace()
    #run this first to get the dictionary of image files to their embeddings
    #embeddings.all_embeddings(config['datapath'])

    #afterwards, run this for linear interpolation
    embeddings.linear_interpolation()