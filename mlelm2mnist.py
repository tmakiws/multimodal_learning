# coding: utf-8
import gzip
import cPickle
from extreme import MLELMClassifier
import numpy as np

def load_mnist():
    f = gzip.open('../Dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def mnist_mlelm(n_hidden=[1000]):
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
    model = MLELMClassifier(n_hidden = n_hidden, n_coef=[0.1, 99999999, 1000], fine_coef=1000000)

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
    mnist_mlelm([700, 700, 15000])
    #mnist_mlelm([700, 700, 15000])
