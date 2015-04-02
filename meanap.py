# -*- coding: utf-8 -*-
import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
    A list of elements that are to be predicted (order doesn't matter)
    predicted : list
    A list of predicted elements (order does matter)
    k : int, optional
    The maximum number of predicted elements
    Returns
    -------
    score : double
    The average precision at k over the input lists
    """
    # print type(predicted)
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
        
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not all(actual):
        return 1.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
    A list of lists of elements that are to be predicted
    (order doesn't matter in the lists)
    predicted : list
    A list of lists of predicted elements
    (order matters in the lists)
    k : int, optional
    The maximum number of predicted elements
    Returns
    -------
    score : double
    The mean average precision at k over the input lists
    """

    res = [apk(a,p,k) for a,p in zip(actual, predicted) if not len(a) == 0]
    
    return np.mean(res)


if __name__ == '__main__':
    actual = [[2,4,6,8,10],[2,4,6,8,10],[2,4,6,8,10]]
    predicted = [[3,2,1,3,8],[4,8,1,2,3],[2,4,8,6,9]]

    print mapk(actual, predicted, k=3)
