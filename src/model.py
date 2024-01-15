import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Model, self).__init__()

        # Use pretrained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final fully connected layer for the number of classes in your dataset
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Lists to store loss values for training and validation
        self.train_losses = []
        self.valid_losses = []

    # Module needs required "forward" function (Forward pass)
    def forward(self, x):
        return self.model(x)

    def train_model(self, Dataset):
        num_epochs=15
        learning_rate=0.00001
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")
            
            # Store metrics for both training and validation phases
            metrics = {
                'train': {'loss': 0.0, 'correct': 0},
                'validation': {'loss': 0.0, 'correct': 0}
            }

            # Train loop
            #In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
            for phase in ['train', 'validation']:
                # As the data loader have the train and valid mode(see data.py)
                if phase == 'train':
                    self.train()
                else:
                    self.eval() 

                for inputs, labels in Dataset.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward only for train phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Update metrics
                    metrics[phase]['loss'] += loss.item() * inputs.size(0)
                    metrics[phase]['correct'] += torch.sum(preds == labels.data)

                # Print the log
                epoch_loss = metrics[phase]['loss'] / Dataset.dataset_sizes[phase]
                epoch_acc = metrics[phase]['correct'].double() / Dataset.dataset_sizes[phase]
                print(f"{phase}_loss: {epoch_loss:.4f}, {phase}_acc: {epoch_acc:.4f}")

            # For plotting
            self.train_losses.append(metrics['train']['loss'] / Dataset.dataset_sizes['train'])
            self.valid_losses.append(metrics['validation']['loss'] / Dataset.dataset_sizes['validation'])

    # Plotting function for training and validation losses
    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs. Train/Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()