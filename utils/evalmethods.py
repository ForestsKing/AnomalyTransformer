import numpy as np
from sklearn.metrics import f1_score

from utils.adjustpredicts import adjust_predicts


def bestf1_threshold(test_scores, test_label, adjust=False, start=0, end=1, search_step=1000):
    best_f1 = 0.0
    best_threshold = 0.0

    for i in range(search_step):
        threshold = start + i * ((end - start) / search_step)
        test_pred = (test_scores > threshold).astype(np.int)
        if adjust:
            test_pred = adjust_predicts(test_label, test_pred)
        f1 = f1_score(test_label, test_pred)

        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1

    return best_threshold
