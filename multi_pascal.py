# coding: utf-8
from extreme import *
from meanap import *
#import cPickle as pkl
#import scipy.sparse as sp
import numpy as np
import sys
from PIL import Image
from boltzmann_machine import BiModalDBM
from sklearn.neighbors import NearestNeighbors

import datetime
# import locale
import os
import errno
import random
from LoadData import load_pascal
from argparse import ArgumentParser


pascal_root = '/home/iwase/dataset/pascal/'

def print_log(f, str):
    f.write(str)  

def standardize(arr):
    arr = np.array(arr)
    mean = np.mean(arr, axis=0)
    arr -= mean
    # print arr.shape, type(arr)
    std = np.std(arr, axis=0)
    # print std, std.shape, arr.T.shape    
    zeros = [0] * arr.shape[0]

    res = np.array([ e / std[i] if std[i] != 0 else zeros for i, e in enumerate(arr.T) ]).T

    return res, mean, std  

def extract_texts(textdb, idx, tags):

    texts = []
    ids = np.where(textdb[idx] == 1)

    if len(ids) > 0:
        for j in ids[0]:
            # print tags[j][0]
            texts.append(tags[j][0])

    return texts

    
def pascal_retrieval(model, nn_k):
  
    print '===================='
    print '   load dataset     '
    print '===================='
    tra_ids, tra_img, tra_txt, tra_lab, val_ids, val_img, val_txt, val_lab, tes_ids, tes_img, tes_txt, tes_lab = load_pascal()
    
    print tra_img.shape, tra_img[100]
    ### test=>train
    #tes_ids, tes_img, tes_txt, tes_lab = tra_ids, tra_img, tra_txt, tra_lab
    #tes_img, tes_txt, tes_lab = tra_img[:100], tra_txt[:100], tra_lab[:100]
    ###

    n_tra = len(tra_img)
    # n_tes = len(tes_img)

    print '===================='
    print '   training         '
    print '===================='
    train_db = model.pre_train(tra_img, tra_txt, filename='train_sharedrep.npy')
    print train_db.shape
    
    # print 'creating test_db'
    # test_db = mae.pre_extraction(unlab_img, None, limit=-1, filename="images.npy")

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

    sum_map, sum_top20 = 0, 0
    
    for t in xrange(0,2):
        if t == 0:
            test_rep = model.pre_extraction(tes_img, None, limit=-1, filename="test_imgquery.npy")
        elif t == 1:
            test_rep = model.pre_extraction(None, tes_txt, limit=-1, filename="test_txtquery.npy")
        elif t == 2:
            test_rep = model.pre_extraction(tes_img, tes_txt, limit=-1, filename="test_sharedrep.npy")

        actual = []
        predicts = []
        prefix = ['img2txt', 'txt2img']
#        tra_idx = [ e+1 for e in xrange(1000) if e % 50 < 40]

        for i, rep in enumerate(test_rep):
            
            #if i == 10:
            #    break
            # print tes_img[i].shape, tes_txt[i].shape
            
            #tes_idx = [ e+1 for e in xrange(1000) if e % 50 > 44]
            tes_class = tes_lab[i]

            f.write("====== result %s (class %s) ======\n" % (i, tes_class))
            
            # tes_texts = extract_texts(tes_txt, i, tags)
            # f.write(" query_text: %s\n" % tes_texts)
            tes_img_no = tes_ids[i]
            pre_img = Image.open(pascal_root + "all_images/%03d.jpg" % tes_img_no)
            pre_img.save(dirname + "/%spas%s.jpg" % (prefix[t], i))
            
            
            #--- find k nearest neighbors
            # nn_ids, dists = find_kneighbors(rep, train_db, nn_k)
            nbrs = NearestNeighbors(n_neighbors=nn_k, algorithm='ball_tree').fit(train_db)
            distances, indices = nbrs.kneighbors([rep])
            nn_ids, dists = indices[0], distances[0]
            
            # nn_ids, dists = find_kneighbors(rep, test_db, nn_k)

            predict = []
            for k, (nn_idx, dist) in enumerate(zip(nn_ids, dists)):

                predict.append(tra_ids[nn_idx])
                tra_class = tra_lab[nn_idx]
                
                # tra_texts = extract_texts(tra_txt, nn_idx, tags)
                if tra_class == tes_class:
                    f.write(" %s: %s (class: %s)\n" % (k, tra_ids[nn_idx], tra_class))
                    f.write(" distance: %s \n" % dist)
                    # f.write(" nearest_images_texts: %s" % tra_texts)
                    f.write("\n")
 
                if k < 10:
                    tra_img_no = tra_ids[nn_idx]
                    res_img = Image.open(pascal_root + "all_images/%03d.jpg" % tra_img_no)
                    res_img.save(dirname + "/%sres%s-%s.jpg" % (prefix[t], i, k))

            actual.append(tra_ids[np.where(tra_lab == tes_class)[0]])
            predicts.append(predict)
            # print i, apk(actual[-1], predict, k=nn_k), actual[-1], predict#, mapk(actual, predicts, k=nn_k)

        
        #print actual, predicts
        map_score = mapk(actual, predicts, k=nn_k)
        print 'map score is %.3f' % map_score
        
        top20_score = mtop20(actual, predicts, k=nn_k)
        print 'top20 score is %.2f' % (top20_score*100)
        
        if t < 2:
            sum_map += map_score
            sum_top20 += top20_score

    f.close()

    return sum_map/2., sum_top20/2. 
 

def ELM_retrieval(args):

    #------ Retrieval Tasks
    print "Retrieval Task"

    joint = args.joint
    hidden = args.hidden
    elm_param = args.reg_param
    cca_param = args.cca_param
    num_layers = args.layers
    fname = args.filename
    quiet = args.quiet

    with open(fname, 'a') as f:
        mae = BiModalStackedELMAE([hidden], [hidden], [], [], [],\
                                  [elm_param], [elm_param], [elm_param], [elm_param], [],\
                                  joint_method=joint, n_comp=128, cca_param=cca_param, iteration=num_layers)
        #mae = BiModalStackedELMAE([1024, dim_bfcca], [1024, dim_bfcca], [], [], [],\
        #                              [elm_param, elm_param], [elm_param, elm_param], [elm_param], [elm_param], [],\
        #                              joint_method="cca", n_comp=64, cca_param=cca_param)

        mean_map, mean_top20 = pascal_retrieval(mae, 50)
        if not quiet:
            print_log(f, str(num_layers) + ' ' + str(hidden) + ' ' + str(elm_param) + ' ' + str(cca_param)
                      + ' mean_map:' + str(mean_map) + ' mean_top20:' + str(mean_top20) + '\n')
        print('\nmean_map:' + str(mean_map))
        print('mean_top20:' + str(mean_top20) + '\n')
        

def HeMap_retrieval(args):

    print "Retrieval Task"

    joint = args.joint
    hidden = args.hidden
    HM_param = args.reg_param
    cca_param = args.cca_param
    num_layers = args.layers
    fname = args.filename
    quiet = args.quiet
    
    with open(fname, 'a') as f:
        hm = BiModalStackedHeMapLayers([hidden], [], [], [HM_param], [], [],\
                                       joint_method=joint, n_comp=64, cca_param=cca_param, iteration=num_layers)
            
        mean_map, mean_top20 = pascal_retrieval(hm, 50)

        if not quiet:
            print_log(f, str(num_layers) + ' ' + str(hidden) + ' ' + str(HM_param) + ' ' + str(cca_param)
                      + ' mean_map:' + str(mean_map) + ' mean_top20:' + str(mean_top20) + '\n')

        print('\nmean_map:' + str(mean_map))
        print('mean_top20:' + str(mean_top20) + '\n')
    
        
if __name__ == "__main__":

    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    # %progは出力できない
    # usage = u'%prog [Args] [Options]\nDetailed options -h or --help'

    parser = ArgumentParser(description=desc)

    parser.add_argument("-j", "--joint", type=str, dest="joint", default='cca',
                      help="joint method", metavar="JOINT_METHOD")
    parser.add_argument("--hidden", type=int, dest="hidden", default=256,
                      help="number of hidden units", metavar="HIDDEN")
    parser.add_argument("-r", "--reg", type=int, dest="reg_param", default=1,
                      help="regularize parameter of feature extraction layers", metavar="REG_PARAM")
    parser.add_argument("-c", "--cca", type=int, dest="cca_param", default=1,
                      help="regularize parameter of cca", metavar="CCA_PARAM")
    parser.add_argument("-l", "--layers", type=int, dest="layers", default=2,
                      help="number of layers", metavar="LAYERS")
    parser.add_argument("-f", "--file", type=str, dest="filename", default='result',
                      help="filename", metavar="FILE")
    parser.add_argument("-q", "--quiet", action='store_true', dest="quiet", default=False, 
                      help="do not print to file")
    args = parser.parse_args()

    #bimodalMLELM(args)
    HeMap_retrieval(args)
