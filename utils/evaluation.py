import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def get_range_proba(predict, label, delay=7):
    predict = np.array(predict)
    label = np.array(label)

    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def reconstruct_series(timestamp, label, predict, delay=7):
    label = reconstruct_label(timestamp, label)
    predict = reconstruct_label(timestamp, predict)
    predict = get_range_proba(predict, label, delay)
    return label.tolist(), predict.tolist()


def calc(pred, true):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pre, gt in zip(pred, true):
        if gt == 1:
            if pre == 1:
                TP += 1
            else:
                FN += 1
        if gt == 0:
            if pre == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


def evaluate_for_all_series(lst_timestamp_label_predict, delay=7, prt=True):
    labels, predicts = [], []
    for timestamp, label, predict, _ in lst_timestamp_label_predict:
        if timestamp == []:
            continue
        lbl, pdt = reconstruct_series(timestamp, label, predict, delay)
        labels += lbl
        predicts += pdt

    f1 = f1_score(labels, predicts)
    pre = precision_score(labels, predicts)
    rec = recall_score(labels, predicts)
    TP, FP, TN, FN = calc(predicts, labels)
    if prt:
        print('precision', pre)
        print('recall', rec)
        print('f1', f1)
        print('-------------------------------')
    return f1, pre, rec, TP, FP, TN, FN
