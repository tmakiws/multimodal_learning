# coding: utf-8
# from extreme import BiModalStackedELMAE
from extreme import BiModalMLELMClassifier
from extreme import BiModalStackedELMAE
import scipy.sparse as sp
import numpy as np
import sys
from PIL import Image

import datetime
# import locale
import os
import errno


flickr_root = '/home/iwase/dataset/flickr/'

def load_flickr25k(n_train=15000, unlab=False):
    
    # labelled images
    image1 = np.load(flickr_root + 'image/labelled/combined-00001-of-00100.npy')
    image2 = np.load(flickr_root + 'image/labelled/combined-00002-of-00100.npy')
    image3 = np.load(flickr_root + 'image/labelled/combined-00003_0-of-00100.npy')
    lab_image = np.concatenate((image1,image2,image3), axis=0)

    # unlabelled images
    if unlab:
        for _, _, fs in os.walk(flickr_root + 'image/unlabelled/'):
            fs.sort()
            for i, f in enumerate(fs):
                print f
                image = np.load(flickr_root + 'image/unlabelled/%s' % f)
                if i == 0:
                    unlab_image = image
                else:
                    if i == 2:
                        break
                    unlab_image = np.concatenate((unlab_image,image), axis=0)
        print unlab_image.shape
    
    else:
        unlab_image = None
        
    # tags(1-of-K) and labels
    text = LoadSparse(flickr_root + 'text/text_all_2000_labelled.npz').todense()
    label = np.load(flickr_root + 'labels.npy')

    # tags(string)
    tags = {}
    with open(flickr_root + "text/vocab.txt") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            spltd = line.split(" ")
            tags[idx] = [spltd[1], spltd[2]]

    
    # train (default: 15k)
    train_image = lab_image[:n_train]
    train_text = np.array(text[:n_train])
    train_label = label[:n_train]
    
    # test (default: 10k)
    test_image = lab_image[n_train:]
    test_text = np.array(text[n_train:])
    test_label = label[n_train:]
    
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


def find_kneighbors(vec, database, k):

    min_dists = []
    nearest_kneighbors = np.array(xrange(k))
    head = 0
    
    for idx, data in enumerate(database):
        dist = np.linalg.norm(vec-data)
        if idx < k:
            min_dists.append(dist)
            if idx == k-1:
                min_dists = np.array(min_dists)

                sorted_i = np.argsort(min_dists)
                nearest_kneighbors = nearest_kneighbors[sorted_i]
                min_dists = min_dists[sorted_i]
                
        if min_dists[head] > dist:
            min_dists[head] = dist
            nearest_kneighbors[head] = idx
            head = (head+1) % k

    sorted_i = np.argsort(min_dists)
    nearest_kneighbors = nearest_kneighbors[sorted_i]
    min_dists = min_dists[sorted_i]            
    
    return list(nearest_kneighbors), list(min_dists)




###################################################
## 1/23(Fri) TODO: Add unlab_img to pre_training ##
###################################################

def flickr_classification(model, unlab=False):

    print '===================='
    print '   load dataset     '
    print '===================='
  
    # convert np.matrix to np.ndarray
    # tra_txt, tes_txt = np.array(tra_txt), np.array(tes_txt)
    if unlab:
        tra_img, tra_txt, tra_lab, tes_img, tes_txt, tes_lab, unlab_img, tags = load_flickr25k(unlab=True)
        tra_img = np.concatenate((tra_img, unlab_img), axis=0)
    else:
        tra_img, tra_txt, tra_lab, tes_img, tes_txt, tes_lab, _, tags = load_flickr25k()
    
    # train
    print '===================='
    print '   training         '
    print '===================='
    model.fit(tra_img, tra_txt, tra_lab)

    # test
    print '===================='
    print '   test             '
    print '===================='
    score = model.score(tes_img, tes_txt, tes_lab, mode='map')
    print 'map score is %.3f %%' % (score*100)


    
def flickr_retrieval(mae, nn_k):
  
    print '===================='
    print '   load dataset     '
    print '===================='
    tra_img, tra_txt, tra_lab, tes_img, tes_txt, tes_lab, unlab_img, tags = load_flickr25k()
    
    n_tra = len(tra_img)
    # n_tes = len(tes_img)

    # convert np.matrix to np.ndarray
    tra_txt, tes_txt = np.array(tra_txt), np.array(tes_txt)
    
 
# 1/26 TODO: Add unlabelled img to training.

    # train
    print '===================='
    print '   training         '
    print '===================='
    mae.pre_train(tra_img, tra_txt)
    # print train_db.shape
    # print train_db[0]
    
    print 'creating test_db'
    print unlab_img.shape
    test_db = mae.pre_extraction(unlab_img, None, limit=-1, filename="images.npy")

    d = datetime.datetime.today()
    now = d.strftime("%Y%m%d_%H%M%S")
    dirname = "retrieval/%s" % now
    
    try:
        os.mkdir(dirname)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise e
        pass

    f = open("retrieval/%s.log" % now,"w")
           
    # test
    print '===================='
    print '   test             '
    print '===================='

    test_rep = mae.pre_extraction(tes_img, tes_txt, limit=-1, filename="tags.npy")

    for i, rep in enumerate(test_rep):

        if i == 10:
            break
        # print tes_img[i].shape, tes_txt[i].shape
    
 
        f.write("====== result %s ======\n" % i)

        tes_texts = extract_texts(tes_txt, i, tags)      
        f.write(" query_text: %s\n" % tes_texts)
        pre_img = Image.open(flickr_root + "images/%s/%s.jpg" \
                                 % ((n_tra+i)/10000, n_tra+i))
        pre_img.save(dirname + "/pre%s.jpg" % i)


        #--- find k nearest neighbors
        nn_ids, dists = find_kneighbors(rep, test_db, nn_k)

        for k, (nn_idx, dist) in enumerate(zip(nn_ids, dists)):

            # tra_texts = extract_texts(tra_txt, nn_idx, tags)

            f.write(" candidate%s: %s \n" % (k, nn_idx))
            f.write(" distance: %s \n" % dist)
            # f.write(" nearest_images_texts: %s" % tra_texts)
            f.write("\n")

            dirno = (nn_idx+5000) / 10000 + 2
            fileno = nn_idx+25000
            res_img = Image.open(flickr_root + "images/%s/%s.jpg" % (dirno, fileno))
            res_img.save(dirname + "/res%s-%s.jpg" % (i, k))

    f.close()


def extract_texts(textdb, idx, tags):

    texts = []
    ids = np.where(textdb[idx] == 1)

    if len(ids) > 0:
        for j in ids[0]:
            # print tags[j][0]
            texts.append(tags[j][0])

    return texts
    
        
if __name__ == "__main__":

    argvs = sys.argv
    argc = len(argvs)
    # print argvs, argc

    #------ Retrieval Tasks
    if argvs[1] == 'r':
        print "Retrieval Task"
        if argc != 3:
            raise Exception("Usage: python multi_flickr.py r")
        elif argvs[2] == 'cca':

            mae = BiModalStackedELMAE([1024,1024], [1024,1024], [],\
                                          [1,1], [1,1], [],\
                                          joint_method="cca", n_comp=16, cca_param=0.1)

        elif argvs[2] == 'pcca':

            mae = BiModalStackedELMAE([1024,1024], [1024,1024], [],\
                                          [1,1], [1,1], [],\
                                          joint_method="pcca", n_comp=16, cca_param=0.1)
        flickr_retrieval(mae, 5)


    #------ Classification Tasks
    elif argvs[1] == 'c':
        print "Classification Task"
        
        if argc != 3 and argc != 4:
            raise Exception("Usage: python multi_flickr.py c "\
                                + "(joint_method: concatenate or cca or pcca)")
        
        elif argvs[2] == 'concatenate':
        
            model = BiModalMLELMClassifier([1024, 1024], [1024,1024], [2048],\
                                               [100, 100], [100, 100], [100],\
                                               fine_coef=1000000)
        elif argvs[2] == 'cca':
        
            n_comp = int(argvs[3])
            model = BiModalMLELMClassifier([1024, 1024], [1024, 1024], [],\
                                               [10, 10], [10, 10], [],\
                                               joint_method="cca", n_comp=n_comp,\
                                               cca_param=0.1, fine_coef=1000)
        elif argvs[2] == 'pcca':
        
            n_comp = int(argvs[3])
            model = BiModalMLELMClassifier([1024, 1024], [1024, 1024], [],\
                                               [100, 100], [100, 100], [],\
                                               joint_method="pcca", n_comp=n_comp,\
                                               cca_param=0.1, fine_coef=1000)

        else:
            raise Exception("Task number is invalid!")
        
        flickr_classification(model)
