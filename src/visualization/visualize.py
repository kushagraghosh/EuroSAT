from visualization.confusion import confusion
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle

def visualize_results(Results,outfolder):
    fig = confusion(Results.y_true,Results.y_pred,Results.classes)
    fig.savefig(outfolder + 'confusion.png', dpi=300)
    
def load_results(path):
    with open(path, 'rb') as handle:
        Results = pickle.load(handle)
    return Results