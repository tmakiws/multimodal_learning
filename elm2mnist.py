# coding: utf-8
import gzip
import cPickle
from extreme import ELMClassifier
import numpy as np
import time
import sys

def load_mnist():
    f = gzip.open('../Dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def mnist_elm(n_hidden=50, domain=[-1., 1.]):
    print "hidden:", n_hidden

    # initialize
    train_set, valid_set, test_set = load_mnist()
    train_data, train_target = train_set
    valid_data, valid_target = valid_set
    test_data, test_target = test_set
    
    # size
    train_size = 50000 # max 50000
    valid_size = 10000 # max 10000
    test_size = 10000 # max 10000

    train_data, train_target = train_data[:train_size], train_target[:train_size]
    valid_data, valid_target = valid_data[:valid_size], valid_target[:valid_size]
    test_data, test_target = test_data[:test_size], test_target[:test_size]

    # train = train + valid
    train_data = np.concatenate((train_data, valid_data))
    train_target = np.concatenate((train_target, valid_target))
    
    # model
    model = ELMClassifier(n_hidden = n_hidden, domain = domain)

    # fit
    #print "fitting ..."
    model.fit(train_data, train_target)

    # test
    print "test score is ",
    score = model.score(test_data, test_target)
    print score

    # valid
#    print "valid score is ",
#    score = model.score(valid_data, valid_target)
#    print score


if __name__ == "__main__":
    time1 = time.clock()
    mnist_elm(15000, [-0.02, 0.02])
    time2 = time.clock()
    time = str(time2-time1)
    print 'time %s s' % time2
    sys.exit()
    # mnist_elm(3000, [-0.02, 0.02])
    # mnist_elm(4000, [-0.02, 0.02])
    # mnist_elm(5000, [-0.02, 0.02])
    # mnist_elm(6000, [-0.02, 0.02])
    # mnist_elm(7000, [-0.02, 0.02])
    # mnist_elm(8000, [-0.02, 0.02])
    # mnist_elm(9000, [-0.02, 0.02])
    # mnist_elm(10000, [-0.02, 0.02])
