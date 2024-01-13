import yaml
import data
import preprocess
import model
import evaluate
from visualization.visualize import load_results, visualize_results
import os

def run(config):
    # Load the data
    print("LOADING DATA")
    Dataset = data.Dataset(config['datapath'])
    Dataset.describe()
    
    # Preprocess the data
    print("PREPROCESSING DATA")
    Dataset = preprocess.preprocess(Dataset)
    Dataset.describe()
    
    # Train the model
    print("TRAINING THE MODEL")
    Model = model.Model()
    Model.train(Dataset)
    
    # Evaluate the model
    print("EVALUATING THE MODEL")
    [prediction, classes] = Model.predict(Dataset)
    Results = evaluate.Results(Dataset.test['y'],prediction, classes)
    Results.print_metrics()
    
    # Save the results to file
    savefile = os.path.join(config['resultspath'],'results.pickle')
    Results.save_metrics(savefile)
    
    # Visualize the results
    print("VISUALIZING RESULTS")
    Loaded_results = load_results(savefile)
    visualize_results(Loaded_results,config['figurepath'])
    

# This is needed to run this file as a script rather than import it as a module
if __name__ == "__main__":

    # Load the configuration file
    with open('config.yaml') as p:
        config = yaml.safe_load(p)
    
    run(config)