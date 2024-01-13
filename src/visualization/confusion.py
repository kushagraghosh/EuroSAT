''' Plot confusion matrices and save to file
'''
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def confusion(y_true,y_pred,labels):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,
                                            labels=labels,
                                            ax=ax,
                                            normalize='true',
                                            cmap='gray')
    return fig
