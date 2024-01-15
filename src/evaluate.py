import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import yaml
import data
import model

class Results:
        
    # Function to evaluate the given model and return Test Accuracy.
    def eval_confusion(self, model, Dataset, datapath):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
        dataloaders, dataset_sizes = Dataset.dataloaders, Dataset.dataset_sizes
        image_datasets = {x: ImageFolder(os.path.join(datapath, x), data_transforms[x]) for x in ['train', 'test']}

        # record the label
        all_labels = []
        all_predictions = []

        # get class names, as the predictions are directly numbers
        class_names = image_datasets['test'].classes
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Append labels and predictions for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        print(f"Accuracy on test data: {100 * correct / total:.2f}%")

        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Display
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized * 100, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix (Percentages)')
        plt.show()

if __name__ == "__main__":
    with open('../config.yaml') as p:
        config = yaml.safe_load(p)

    Dataset = data.Dataset(config['datapath'])

    Model = model.Model()

    # Load the saved model
    Model.load_state_dict(torch.load('trained_resnet_model.pth'))

    Results = Results()
    Results.eval_confusion(Model, Dataset, config['datapath'])