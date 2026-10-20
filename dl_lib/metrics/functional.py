import numpy as np

def check_threshold(p_i, threshold):
    return int(p_i >= threshold) 

def apply_threshold(p, threshold):
    check_vectorized = np.vectorize(check_threshold)
    return check_vectorized(p, threshold);

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

    return cm["tp"] / (cm["tp"] + cm["fp"])

def recall(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    return cm["tp"] / (cm["tp"]+cm["fn"])

def f1_score(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true)

    return 2 * cm["tp"] / (2 * cm["tp"] + cm["fn"] + cm["fp"])




