"""
This is a DIY module providing some basic tools in classification applications. The function in this module tries to compensate some trivial but missing functions in sklearn package. 
"""
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score

def classification_metrics(ytrue, ypred):
    """
    DIY function to return binary classification accuracy, precision (PPR), sensitivity (recall, TPR), and specificity (TNR)

    work for both (1, 0) coding and (1, -1) coding
    """
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)   # both predict value and true value are one-dimensional vectors
    p_count = np.sum(ytrue > 0)
    n_count = ytrue.shape[0] - p_count
    a = accuracy_score(ytrue, ypred)
    b = precision_score(ytrue, ypred)
    c = recall_score(ytrue, ypred)
    d = (a * ytrue.shape[0] - c * p_count) / n_count
    return a, b, c, d
