# -*- coding: utf-8 -*-
import gzip
import cPickle
from extreme import StackedELMAutoEncoder
import scipy.sparse as sp
import numpy as np
from extreme import sigmoid


def read_integers():
    return map(int, raw_input().split())

if __name__ == '__main__':
    
