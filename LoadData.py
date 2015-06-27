import cPickle as pkl
import scipy.sparse as sp
import numpy as np
import os
import random

def load_pascal(n_train=800, unlab=False):

    pascal_root = '/home/iwase/dataset/pascal/'  
    phow = np.load(pascal_root + 'pascal_phow.npy')
    gist = np.load(pascal_root + 'pascal_gist.npy')
    mpeg7 = np.load(pascal_root + 'pascal_mpeg7.npy')
    texts = np.load(pascal_root + 'pascal_bow.npy')
    labels = [ e/50 + 1 for e in range(1000)]

    images = np.hstack((phow, gist, mpeg7))

#    all_no = []
    train_image, train_text, train_label, train_ids = [], [], [], []
    val_image, val_text, val_label, val_ids = [], [], [], []
    test_image, test_text, test_label, test_ids = [], [], [], []
#    tra100_image, tra100_text, tra100_label = [], [], []

    random.seed(1)
    all_data = zip(range(1000), images, texts, labels)

    # extract 5 cases for validation and 5 cases for test respectively from each category
    for c in xrange(20):
        data_c = random.sample(all_data[50*c:50*(c+1)], 50)
        #print type(data_c[:5]), data_c[:1]
        for i in xrange(len(data_c)):
            if i % 50 < 40:
                train_ids.append(data_c[i][0])
                train_image.append(data_c[i][1])
                train_text.append(data_c[i][2])
                train_label.append(data_c[i][3])
            elif i % 50 < 45:
                val_ids.append(data_c[i][0])
                val_image.append(data_c[i][1])
                val_text.append(data_c[i][2])
                val_label.append(data_c[i][3])
            else:
                test_ids.append(data_c[i][0])
                test_image.append(data_c[i][1])
                test_text.append(data_c[i][2])
                test_label.append(data_c[i][3])


    print np.array(train_image).shape
    print np.array(test_image).shape
    print np.array(test_text).shape

    return np.array(train_ids), np.array(train_image), np.array(train_text), np.array(train_label),\
        np.array(val_ids), np.array(val_image), np.array(val_text), np.array(val_label),\
        np.array(test_ids), np.array(test_image), np.array(test_text), np.array(test_label)


def load_flickr25k(n_train=15000, unlab=False):
    
    image1 = np.load(flickr_root + 'image/labelled/combined-00001-of-00100.npy')
    image2 = np.load(flickr_root + 'image/labelled/combined-00002-of-00100.npy')
    image3 = np.load(flickr_root + 'image/labelled/combined-00003_0-of-00100.npy')


    if unlab:
        for _, _, fs in os.walk(flickr_root + 'image/unlabelled/'):
            fs.sort()
            for i, f in enumerate(fs):
                image = np.load(flickr_root + 'image/unlabelled/%s' % f)
                if i == 0:
                    unlab_image = image
                else:
                    if i == 2:
                        break
                    unlab_image = np.concatenate((unlab_image,image), axis=0)
                print f
                        
        print "unlabelled image: %s" % str(unlab_image.shape)
        
        # unlab_image = standardize(unlab_image)

    else:
        unlab_image = None
            
        
    text = LoadSparse(flickr_root + 'text/text_all_2000_labelled.npz').todense()
    label = np.load(flickr_root + 'labels.npy')
    lab_image = np.concatenate((image1,image2,image3), axis=0)

    tags = {}
    with open(flickr_root + "text/vocab.txt") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            spltd = line.split(" ")
            tags[idx] = [spltd[1], spltd[2]]

    
    # train (default: 15k)
    train_image = np.array(lab_image[:n_train])
    train_text = np.array(text[:n_train])
    train_label = np.array(label[:n_train])
    
    # test (default: 10k)
    test_image = np.array(lab_image[n_train:])
    test_text = np.array(text[n_train:])
    test_label = np.array(label[n_train:])

    # np.save("tra_txt.npy", train_text)

    # standardize training data
    # train_image, train_image_mean, train_image_std = standardize(train_image)
    # train_text, train_text_mean, train_text_std = standardize(train_text)
    #train_text = np.array([ txt*5 / sum(txt) if sum(txt) > 0 else txt for txt in train_text ])
    # print np.mean(train_image), np.std(train_image)

    # test
    # test_image = (test_image - train_image_mean) / train_image_std
    # test_text = (test_text - train_text_mean) / train_text_std
    #test_text = np.array([ txt*5 / sum(txt) for txt in test_text])

    return train_image, train_text, train_label,\
        test_image, test_text, test_label,\
        unlab_image, tags

    
def LoadSparse(inputfile, verbose=False):
    """Loads a sparse matrix stored as npz file."""
    npzfile = np.load(inputfile)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                         npzfile['indptr']),
                        shape=tuple(list(npzfile['shape'])))
    if verbose:
      print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                          mat.shape.__str__())
    return mat


# Use find_kneighbors(mae, 1) alternatively.
#
# def find_neighbor(vec, database):

#     min_dist = 10**5
#     nearest_neighbor = None
    
#     for idx, data in enumerate(database):
#         dist = np.linalg.norm(vec-data)
#         if min_dist > dist:
#             min_dist = dist
#             nearest_neighbor = idx

#     return nearest_neighbor, min_dist


def load_pretrained_model(filename):
    with open(filename) as f:
        print "loading pretrained model"
        return pkl.load(f)
