import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_for_all_series(labels, predicts, prt = True):

    label = labels.values
    predicts = predicts.values
    f1 = f1_score(labels, predicts)
    pre = precision_score(labels, predicts)
    rec = recall_score(labels, predicts)
    if prt:
        print('precision|', pre)
        print('recall   |', rec)
        print('f1       |', f1)
        print('-------------------------------')
    return f1, pre, rec
