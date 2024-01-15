import yaml
import data
import model
import evaluate
import os
import torch

def run(config):
    # Load the data
    print("LOADING DATA")
    Dataset = data.Dataset(config['datapath'])
    Dataset.describe()
    
    # Train the model
    print("TRAINING THE MODEL")
    Model = model.Model()
    Model.train_model(Dataset)
    Model.plot_losses()

    # save the model
    torch.save(Model.state_dict(), 'trained_resnet_model.pth')

    # Evaluate the model
    print("EVALUATING THE MODEL")
    Results = evaluate.Results()
    Results.eval_confusion(Model, Dataset, config['datapath'])


# This is needed to run this file as a script rather than import it as a module
if __name__ == "__main__":

    # Load the configuration file
    with open('../config.yaml') as p:
        config = yaml.safe_load(p)
    
    run(config)