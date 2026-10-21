import numpy as np


def apply_threshold(p, threshold):
    return (p >= threshold).astype(int)

def confusion_matrix(y_pred, y_true):

    dict = {
        "tp":np.sum((y_pred == 1) & (y_true == 1)),
        "tn":np.sum((y_pred == 0) & (y_true == 0)),
        "fp":np.sum((y_pred == 1) & (y_true == 0)),
        "fn":np.sum((y_pred == 0) & (y_true == 1)),
    }

    return dict

def accuracy(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    return (cm["tp"] + cm["tn"]) / np.size(y_pred)

def precision(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    if(cm["tp"] + cm["fp"] == 0):
        return 0.

    return cm["tp"] / (cm["tp"] + cm["fp"])

def recall(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    if(cm["tp"] + cm["fn"] == 0):
        return 0.

    return cm["tp"] / (cm["tp"]+cm["fn"])

def f1_score(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    if((2 * cm["tp"] + cm["fn"] + cm["fp"]) == 0): 
        return 0.

    return 2 * cm["tp"] / (2 * cm["tp"] + cm["fn"] + cm["fp"])




