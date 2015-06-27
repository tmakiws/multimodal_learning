# coding: utf-8
# online learning

import sys
import numpy as np
import cPickle as pkl
import meanap
# from sklearn.cross_decomposition import CCA
from layers import *
from mycca import MyCCA
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2


class BiModalMLELMClassifier(object):
    """
    Bi-Modal Multi-Layer Extreme Learning Machine Classifier

    
    """
    def __init__(self, n_hidden1, n_hidden2, n_joint,
                 n_coef1, n_coef2, n_coefj,
                 joint_method="concatenate", n_comp=None,
                 cca_param=None, fine_coef=1000., activation=sigmoid):

        self.fine_coef = fine_coef
        self.mae = BiModalStackedELMAE(n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_joint,
                                       n_coef1, n_coef2, n_coef3, n_coef4, n_coefj,
                                       joint_method=joint_method, n_comp=n_comp,
                                       cca_param=cca_param, activation=activation)
        
    def fine_tune(self, teacher):

        print '== fine_tune'
        # get data for fine_tune
        sys.stdout.write("\r  data for fine_tune")
        sys.stdout.flush()
        H = np.array(self.mae.data4fine)
        # data = self.activation(self.data4fine)
        signal = self.signal
        print " done."
        
        # coefficient of regularization for fine_tune
        sys.stdout.write("\r  coefficient")
        sys.stdout.flush()
        np_id = np.identity(min(H.shape))
        if self.fine_coef == 0:
            coefficient = 0
        else:
            coefficient = 1. / self.fine_coef
        print " done."
        
        # pseudo inverse
        sys.stdout.write("\r  pseudo inverse")
        sys.stdout.flush()
        regular = coefficient * np_id
        if H.shape[0] < H.shape[1]:
            Hp = np.linalg.inv(np.dot(H, H.T) + regular)
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.linalg.inv(np.dot(H.T, H) + regular)
            Hp = np.dot(Hp, H.T)
        print " done."

        # set beta for fine_tune
        sys.stdout.write("\r  set beta")
        sys.stdout.flush()
        beta = np.dot(Hp, np.array(signal))
        self.fine_beta = beta
        print " done."
        

    def fine_extraction(self, data):
        #print "fine_extraction"
        #print "data:", np.array(data).shape
        #print "beta:", np.array(self.fine_beta).shape
        return np.dot(data, self.fine_beta)

    def create_signal(self, input1, input2, teacher):
        # initialize classes
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        
        # initialize signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])
        self.signal = signal
        
    
    def fit(self, input1, input2, teacher):

        if isinstance(teacher, list) or isinstance(teacher, np.ndarray):
            self.signal = teacher
            self.classes = range(len(self.signal))
        elif isinstance(teacher, int) or isinstance(teacher, str):
            self.create_signal(input1, input2, teacher)
        else:
            raise Exception("type of teacher is not acceptable.")            
        
        # pre_train fine_tune
        self.mae.pre_train(input1, input2)
        self.fine_tune(teacher)

    def predict(self, input1, input2, limit=1):
        # get predict_output
        hidden = self.mae.pre_extraction(input1, input2)
        output = self.fine_extraction(hidden)
        predict_output = np.array(output)

        # get predict_classes ranking
        predict_classes = []
        for out in predict_output:
            ordered = sorted(enumerate(out), key=lambda x: -x[1])
            ranking = []
            for i, (idx, pr) in enumerate(ordered):
                if i == limit:
                    break
                ranking.append(self.classes[idx])

            # predict_classes.append(self.classes[np.argmax(out)])
            predict_classes.append(ranking)
                
        return predict_classes

    def save(self, name):
        with open(name, 'wb') as op:
            pkl.dump(self, op)

    def score(self, input1, input2, teacher, mode='default'):

        if mode == 'default':
            # get score
            count = 0
            length = len(teacher)
            predicted_classes = self.predict(input1, input2)
            for i in xrange(length):
                if predicted_classes[i][0] == teacher[i]:
                    count += 1

            return count * 1.0 / length

        elif mode == 'map':
            n_classes = len(self.classes)
            predicted_classes = self.predict(input1, input2, limit=n_classes)
            # print self.signal, type(self.signal)
            actual = [[ i for i, e in enumerate(arr) if e > 0 ] for arr in self.signal]

            return meanap.mapk(actual, predicted_classes, k=n_classes)

    
class BiModalStackedELMAE(object):

    def __init__(self, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_joint,
                 n_coef1, n_coef2, n_coef3, n_coef4, n_coefj,
                 joint_method="concatenate", n_comp=None, cca_param=None, activation=sigmoid,
                 iteration=2, dim_add=-1):
        
        modal1, modal2, cca = [], [], []
        self.iteration = int(iteration)

        for i in xrange(self.iteration):
            modal1.append(StackedELMAutoEncoder(n_hidden1, n_coef1, activation=activation))
            modal2.append(StackedELMAutoEncoder(n_hidden1, n_coef2, activation=activation))
            # Create CCA Object
            if joint_method == "cca" or joint_method == "pcca":
                if n_comp == None:
                    raise Exception("require n_comp if joint_method is 'cca' or 'pcca'.")
                else:
                    # self.cca = CCA(n_components=n_comp)
                    cca.append(MyCCA(reg_param=cca_param, show_runtime=True))
                    #self.cca = MyCCA(reg_param=cca_param, show_runtime=True)
            else:
                pass
            

        self.modal1 = np.array(modal1)
        self.modal2 = np.array(modal2)
        self.cca = np.array(cca) 

        #self.modal1 = StackedELMAutoEncoder(n_hidden1, n_coef1, activation=activation)
        #self.modal2 = StackedELMAutoEncoder(n_hidden2, n_coef2, activation=activation)
        
        #self.modal1_2 = StackedELMAutoEncoder(n_hidden3, n_coef3, activation=activation)
        #self.modal2_2 = StackedELMAutoEncoder(n_hidden4, n_coef4, activation=activation)

        self.joint_layers = StackedELMAutoEncoder(n_joint, n_coefj, activation=activation)
        self.joint_method = joint_method
        self.n_comp = n_comp
        self.dim_add = dim_add
        

        # Create CCA Object for joint
        if joint_method == "cca" or joint_method == "pcca":
            if n_comp == None:
                raise Exception("require n_comp if joint_method is 'cca' or 'pcca'.")
            else:
                # self.cca = CCA(n_components=n_comp)
                self.cca_j = MyCCA(n_components=n_comp, reg_param=cca_param, show_runtime=True)
        else:
            self.cca_j = None

        print self.cca_j.get_params()


    def crossmodal_training(self, input1, input2, ccaobj):

        if len(input1) != len(input2):
            d = min(len(input1), len(input2))
            input1 = input1[:d]
            input2 = input2[:d]
        else:
            d = len(input1)
        
        print "Dimension: %s" % d
        
        # np.save("modal1_out.npy", input1)
        # np.save("modal2_out.npy", input2)

        print "dimension reduction"
        #------  sklearn cca library ( without regularization term )
        # x_c, y_c = cca.fit_transform(self.modal1.data4fine, self.modal2.data4fine)
        
        #------ my cca code ( with regularization term )
        x_c, y_c = ccaobj.fit_transform(input1, input2)
        
        corr = []
        for i in xrange(16):
            corr.append(np.corrcoef(x_c[:,i], y_c[:,i])[0,1])

        print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)                            

        #print  self.cca.eigvals[:512]

        #h_out1 = x_c
        #h_out2 = y_c
        h_out1 = sigmoid(x_c)
        h_out2 = sigmoid(y_c)
        
        # plt.plot(x_c[:,0], x_c[:,1], "ro")
        # plt.plot(y_c[:,0], y_c[:,1], "bo")
        # plt.plot(h_out[:,0], h_out[:,1], "yo")
        # plt.show()

        # np.save("cross_out.npy", h_out)
        
        return h_out1, h_out2
    


    def joint(self, input1, input2, method='pcca'):
        
        if len(input1) != len(input2):
            d = min(len(input1), len(input2))
            input1 = input1[:d]
            input2 = input2[:d]
        else:
            d = len(input1)
        
        print "Dimension: %s" % d
        
        # np.save("modal1_out.npy", input1)
        # np.save("modal2_out.npy", input2)

        if method == "concatenate":
            h_out = np.concatenate((input1, input2), axis=1)
            # print self.modal1.data4fine.shape, h_ott.shape

        elif method == "cca":

            print "dimension reduction to %s" % self.n_comp
            #------  sklearn cca library ( without regularization term )
            # x_c, y_c = cca.fit_transform(self.modal1.data4fine, self.modal2.data4fine)

            #------ my cca code ( with regularization term )
            x_c, y_c = self.cca_j.fit_transform(input1, input2)
            print x_c.shape
            corr = []
            for i in xrange(x_c.shape[1]):
                corr.append(np.corrcoef(x_c[:,i], y_c[:,i])[0,1])

            h_out = (x_c + y_c) / 2.
            
            print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)

            #plt.plot(x_c[:,0], x_c[:,1], "ro")
            #plt.plot(y_c[:,0], y_c[:,1], "bo")
            #plt.plot(h_out[:,0], h_out[:,1], "yo")
            #plt.show()

        elif method == "pcca":

            print "dimension reduction to %s" % self.n_comp
            #------ my pcca code ( with regularization term )
            h_out = self.cca_j.fit_ptransform(input1, input2)
            print "eigvals: " , self.cca_j.eigvals
            #plt.plot(h_out[:,0], h_out[:,1], "ro")
            #plt.show()

        else:
            raise Exception("joint value %s is invalid. Set 'concatenate' or 'cca' or 'pcca'." % method)

        np.save("joint_out.npy", h_out)
        return h_out
            

    def pre_train(self, input1, input2, filename=None):

        hiddens1, hiddens2 = np.random.random((input1.shape[0], 0)), np.random((input2.shape[0], 0))

        for i in xrange(self.iteration):
            
            if i != 0:
                print input1.shape, h_out1.shape, h_out1[:,:self.dim_add].shape
                #input1 = np.concatenate((input1, h_out1[:,:self.dim_add]), axis=1)
                #input2 = np.concatenate((input2, h_out2[:,:self.dim_add]), axis=1)
                input1 = h_out1#[:,:self.dim_add]
                input2 = h_out2#[:,:self.dim_add]
            
            print '== modal1 stacked auto encoder'
            h_out1 = self.modal1[i].pre_train(input1)

            print '== modal2 stacked auto encoder'
            h_out2 = self.modal2[i].pre_train(input2)

            if i != self.iteration-1:
                print "== crossmodal training"
                print "Crossmodal learning method: cca"
                h_out1, h_out2 = self.crossmodal_training(h_out1, h_out2, self.cca[i])
                
            hiddens1 = np.hstack((hiddens1, h_out1))
            hiddens2 = np.hstack((hiddens2, h_out2))


                
        #print '== modal1 stacked auto encoder'
        #h_out1 = self.modal1_2.pre_train(h_out1)
        #np.save('modal1_tra.npy', h_out1)

        #print '== modal2 stacked auto encoder'
        #h_out2 = self.modal2_2.pre_train(h_out2)
        #np.save('modal2_tra.npy', h_out2)

        print "== Joint"
        print "Joint Method: %s" % self.joint_method
        h_out = self.joint(h_out1, h_out2, method=self.joint_method)
        np.save('joint_tra.npy', h_out)

        print '== classification layers'
        self.joint_layers.pre_train(h_out)
        shared_rep = self.joint_layers.pre_extraction(h_out)
        
        # set betas and data for fine_tune
        self.data4fine = shared_rep
        if filename:
            np.save(filename, self.data4fine)

        return shared_rep

    
    def pre_extraction(self, input1, input2, limit=-1, filename=None):

        hiddens1, hiddens2 = np.random.random((input1.shape[0], 0)), np.random((input2.shape[0], 0))

        if input1 != None or input2 != None:
            
            for i in xrange(self.iteration):

                if i != 0:
                    #input1 = np.concatenate((input1, h_out1[:,:self.dim_add]), axis=1)
                    #input2 = np.concatenate((input2, h_out2[:,:self.dim_add]), axis=1)
                    if input1 != None:
                        input1 = h_out1#[:,:self.dim_add]
                    if input2 != None:                        
                        input2 = h_out2#[:,:self.dim_add]
                    
                h_out1 = self.modal1[i].pre_extraction(input1)
                h_out2 = self.modal2[i].pre_extraction(input2)
                if h_out1 != None:
                    print h_out1.shape
                if h_out2 != None:
                    print h_out2.shape


                if i != self.iteration-1:
                    h_out1, h_out2 = self.cca[i].transform(h_out1, h_out2)
                    corr = []
                    
                    if input1 != None and input2 != None:

                        for i in xrange(self.n_comp):
                            corr.append(np.corrcoef(h_out1[:,i], h_out2[:,i])[0,1])
                            
                        print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)
                    
                if h_out1 != None:
                    h_out1 = sigmoid(h_out1)
                    hiddens1 = np.hstack((hiddens1, h_out1))
                if h_out2 != None:
                    h_out2 = sigmoid(h_out2)
                    hiddens2 = np.hstack((hiddens2, h_out2))
                    

            #h_out1 = self.modal1_2.pre_extraction(h_out1)
            #h_out2 = self.modal2_2.pre_extraction(h_out2)
            np.save('modal1_tes.npy', h_out1)
            np.save('modal2_tes.npy', h_out2)
        
            if self.joint_method == "concatenate":
                h_out = np.concatenate((h_out1, h_out2), axis=1)
                # print self.modal1.data4fine.shape, h_out.shape
                
            elif self.joint_method == "cca":
                #------ my cca code ( with regularization term )
                x_c, y_c = self.cca_j.transform(h_out1, h_out2)         
                if x_c != None and y_c != None:
                    h_out = (x_c + y_c) / 2.

                    corr = []
                    for i in xrange(self.n_comp):
                        corr.append(np.corrcoef(x_c[:,i], y_c[:,i])[0,1])
                    print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)

                elif x_c != None:
                    h_out = x_c
                else:
                    h_out = y_c


            elif self.joint_method == "pcca":
                #------ my pcca code ( with regularization term )
                h_out = self.cca_j.ptransform(h_out1, h_out2)         

            np.save('joint_tes.npy', h_out)
                
        
        else:
            raise Exception("Inputs are all None.")

        if filename:
            np.save(filename, h_out)

        shared_rep = self.joint_layers.pre_extraction(h_out)
        return shared_rep


class BiModalStackedHeMapLayers(BiModalStackedELMAE): 
    
    def __init__(self, n_hidden1, n_hidden2 , n_joint,
                 n_coef1, n_coef2, n_coefj,
                 joint_method="concatenate", n_comp=None, cca_param=None, activation=sigmoid,
                 iteration=2, dim_add=-1):
        
        hemaps, cca = [], []
        self.iteration = int(iteration)

        for i in xrange(self.iteration):
            hemaps.append(StackedHeMap(n_hidden1, n_coef1, activation=activation))

            # Create CCA Object
            if joint_method == "cca" or joint_method == "pcca":
                if n_comp == None:
                    raise Exception("require n_comp if joint_method is 'cca' or 'pcca'.")
                else:
                    # self.cca = CCA(n_components=n_comp)
                    cca.append(MyCCA(reg_param=cca_param, show_runtime=True))
                    #self.cca = MyCCA(reg_param=cca_param, show_runtime=True)
            else:
                pass
            

        self.hemaps = np.array(hemaps)
        #self.modal2 = np.array(modal2)
        self.cca = np.array(cca) 

        #self.hemap_2 = StackedHeMap(n_hidden2, n_coef2, activation=activation)

        self.joint_layers = StackedELMAutoEncoder(n_joint, n_coefj, activation=activation)
        self.joint_method = joint_method
        self.n_comp = n_comp
        self.dim_add = dim_add
        

        # Create CCA Object for joint
        if joint_method == "cca" or joint_method == "pcca":
            if n_comp == None:
                raise Exception("require n_comp if joint_method is 'cca' or 'pcca'.")
            else:
                self.cca_j = MyCCA(n_components=n_comp, reg_param=cca_param, show_runtime=True)
        else:
            self.cca_j = None

        print self.cca_j.get_params()


    def pre_train(self, input1, input2, filename=None):

        hiddens1, hiddens2 = np.random.random((input1.shape[0], 0)), np.random.random((input2.shape[0], 0))

        for i in xrange(self.iteration):
            
            if i != 0:
                print input1.shape, h_out1.shape, h_out1[:,:self.dim_add].shape
                input1 = np.concatenate((input1, h_out1[:,:self.dim_add]), axis=1)
                input2 = np.concatenate((input2, h_out2[:,:self.dim_add]), axis=1)
                #input1 = h_out1#[:,:self.dim_add]
                #input2 = h_out2#[:,:self.dim_add]
            
            print '== stacked HeMapLayers'
            h_out1, h_out2 = self.hemaps[i].pre_train(input1, input2)

            if i != self.iteration-1:
                print "== crossmodal training"
                print "Crossmodal learning method: cca"
                h_out1, h_out2 = self.crossmodal_training(h_out1, h_out2, self.cca[i])

                print 'hiddens1.shape, h_out1.shape',  hiddens1.shape, h_out1.shape 

            hiddens1 = np.hstack((hiddens1, h_out1))
            hiddens2 = np.hstack((hiddens2, h_out2))

                

        #print '== hemap stacked auto encoder'
        #h_out = self.hemap_2.pre_train(h_out1, h_out2)
        np.save('hemap1.npy', h_out1)
        np.save('hemap2.npy', h_out2)

        print "== Joint"
        print "Joint Method: %s" % self.joint_method

        #20150618
        h_out1, h_out2 = np.array(hiddens1), np.array(hiddens2)
        print hiddens1.shape, hiddens2.shape

        h_out = self.joint(h_out1, h_out2, method=self.joint_method)
        #h_out = (h_out1 + h_out2)/2.
        np.save('joint_tra.npy', h_out)

        print '== classification layers'
        self.joint_layers.pre_train(h_out)
        shared_rep = self.joint_layers.pre_extraction(h_out)
        
        # set betas and data for fine_tune
        self.data4fine = shared_rep
        if filename:
            np.save(filename, self.data4fine)

        return shared_rep

    
    def pre_extraction(self, input1, input2, limit=-1, filename=None):

        if input1 != None:
            hiddens1 = np.random.random((input1.shape[0], 0))

        if input2 != None:
            hiddens2 = np.random.random((input2.shape[0], 0))


        if input1 != None or input2 != None:           
            
            for i in xrange(self.iteration):

                if i != 0:
                    if input1 != None:
                        input1 = np.concatenate((input1, h_out1[:,:self.dim_add]), axis=1)
                        #input1 = h_out1[:,:self.dim_add]
                    if input2 != None:                        
                        input2 = np.concatenate((input2, h_out2[:,:self.dim_add]), axis=1)
                        #input2 = h_out2[:,:self.dim_add]
                    
                h_out1, h_out2 = self.hemaps[i].pre_extraction(input1, input2)

                if h_out1 != None:
                    print h_out1.shape
                if h_out2 != None:
                    print h_out2.shape


                if i != self.iteration-1:
                    h_out1, h_out2 = self.cca[i].transform(h_out1, h_out2)
                    corr = []
                    
                    if input1 != None and input2 != None:

                        for i in xrange(self.n_comp):
                            corr.append(np.corrcoef(h_out1[:,i], h_out2[:,i])[0,1])
                            
                        print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)
                    
                    
                if h_out1 != None:
                    h_out1 = sigmoid(h_out1)
                    print hiddens1.shape, h_out1.shape
                    hiddens1 = np.hstack((hiddens1, h_out1))
                if h_out2 != None:
                    h_out2 = sigmoid(h_out2)
                    hiddens2 = np.hstack((hiddens2, h_out2))


            #h_out1, h_out2 = self.hemap_2.pre_extraction(h_out1, h_out2)

            np.save('modal1_tes.npy', h_out1)
            np.save('modal2_tes.npy', h_out2)
        
            if self.joint_method == "concatenate":
                h_out = np.concatenate((h_out1, h_out2), axis=1)
                # print self.modal1.data4fine.shape, h_out.shape
                
            elif self.joint_method == "cca":
                #------ my cca code ( with regularization term )
                

                #20150618
                if input1 != None:
                    h_out1 = hiddens1
                if input2 != None:
                    h_out2 = hiddens2
                
                x_c, y_c = self.cca_j.transform(h_out1, h_out2)         
                #x_c, y_c = h_out1, h_out2
                if x_c != None and y_c != None:
                    h_out = (x_c + y_c) / 2.
                    corr = []
                    for i in xrange(self.n_comp):
                        corr.append(np.corrcoef(x_c[:,i], y_c[:,i])[0,1])
                    print np.max(np.array(corr)), np.min(np.array(corr)), np.mean(np.array(corr)), np.array(corr)

                elif x_c != None:
                    h_out = x_c
                else:
                    h_out = y_c
                


            elif self.joint_method == "pcca":
                #------ my pcca code ( with regularization term )
                h_out = self.cca_j.ptransform(h_out1, h_out2)         

            np.save('joint_tes.npy', h_out)
                       
        else:
            raise Exception("Inputs are all None.")

        if filename:
            np.save(filename, h_out)

        shared_rep = self.joint_layers.pre_extraction(h_out)
        return shared_rep
    
    
class MLELMClassifier(object):
    """
    Multi-Layer Extreme Learning Machine Classifier

    
    """
    
    def __init__(self, n_hidden=None, n_coef=None,
                 fine_coef=1000., activation=sigmoid):

        self.fine_coef = fine_coef
        self.sae = StackedELMAutoEncoder(n_hidden, n_coef, activation)


        
    def fine_tune(self, teacher):
        print "fine_tune"
        # get data for fine_tune
        
        #0427
        #sys.stdout.write("\r  data for fine_tune")
        #sys.stdout.flush()
        
        H = np.array(self.sae.data4fine)
        # data = self.activation(self.data4fine)
        signal = self.signal
        #print " done."
        
        # coefficient of regularization for fine_tune
        #sys.stdout.write("\r  coefficient")
        #sys.stdout.flush()
        np_id = np.identity(min(H.shape))
        if self.fine_coef == 0:
            coefficient = 0
        else:
            coefficient = 1. / self.fine_coef
        #print " done."
        
        # pseudo inverse
        #sys.stdout.write("\r  pseudo inverse")
        #sys.stdout.flush()
        regular = coefficient * np_id
        Hp = np.linalg.pinv(H)
        # if H.shape[0] < H.shape[1]:
        #     Hp = np.linalg.inv(np.dot(H, H.T) + regular)
        #     Hp = np.dot(H.T, Hp)
        # else:
        #     Hp = np.linalg.inv(np.dot(H.T, H) + regular)
        #     Hp = np.dot(Hp, H.T)
        #print " done."

        # set beta for fine_tune
        #sys.stdout.write("\r  set beta")
        #sys.stdout.flush()
        beta = np.dot(Hp, np.array(signal))
        self.fine_beta = beta
        #print " done."
        

    def fine_extraction(self, data):
        #print "fine_extraction"
        #print "data:", np.array(data).shape
        #print "beta:", np.array(self.fine_beta).shape
        return np.dot(data, self.fine_beta)
        
    def fit(self, input, teacher):
        # initialize classes
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)
        
        # initialize signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])
        self.signal = signal
            
        # pre_train fine_tune
        self.sae.pre_train(input)
        self.fine_tune(teacher)

    def predict(self, input):
        # get predict_output
        hidden = self.sae.pre_extraction(input)
        output = self.fine_extraction(hidden)
        predict_output = np.array(output)

        # get predict_classes from index of max_function(predict_output)
        predict_classes = []
        for o in predict_output:
            predict_classes.append(self.classes[np.argmax(o)])

        return predict_classes
    
    def score(self, input, teacher):
        # get score
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]:
                count += 1
        return count * 1.0 / length


class StackedHeMap(object):
    
    def __init__(self, n_hiddens, lambs, activation=sigmoid):
        
        self.n_hiddens = n_hiddens
        if lambs != None:
            self.lambs = np.array(lambs)
        else:
            self.lambs = np.array([1] * len(n_hiddens))
        
        HeMaps = []
        for i, (n_hidden, lamb) in enumerate(zip(self.n_hiddens, self.lambs)):
            hm = HeMapLayer(n_hidden=n_hidden, lamb=lamb, activation=activation)
            HeMaps.append(hm)
        self.HeMaps = HeMaps


    def pre_train(self, input1, input2):
    
        print "pre_train"
        data1, data2 = input1, input2

        for i, hm in enumerate(self.HeMaps):
            print " ", i,"hm fit"
            data1, data2 = hm.fit(data1, data2)

        return data1, data2

        
    def pre_extraction(self, input1, input2):
        
        print 'pre_extraction'
        data1, data2 = input1, input2
        
        for i, hm in enumerate(self.HeMaps):
            data1, data2 = hm.fprop(data1, data2)

        return data1, data2


##########################################################
##  Stacked Extreme Learning Machine AutoEncoder
##########################################################
   
class StackedELMAutoEncoder(object):
    """
    Stacked Extreme Learning Machine AutoEncoder

    
    """
    
    def __init__(self, n_hidden=None, n_coef=None,
                 activation=sigmoid, seed=123):
        # initialize size of neuron
        if n_hidden is None:
            raise Exception("nlist_hidden is udefined")
        self.n_hidden = n_hidden
        if n_coef is None:
            n_coef = [None] * len(n_hidden)
        self.coef = n_coef
        self.activation = activation

        # initialize auto_encoder
        auto_encoders = []
        for i, num in enumerate(n_hidden):
            ae = ELMAutoEncoder(activation=activation, n_hidden=num, coef=n_coef[i], seed=seed)
            auto_encoders.append(ae)
        self.auto_encoders = auto_encoders

    def pre_train(self, input):

        # pre_train
        print "pre_train"
        data = input
        betas = []
        for i, ae in enumerate(self.auto_encoders):
            # fit auto_encoder
            print " ", i,"ae fit"
            ae.fit(data)

            # get beta
            beta = ae.layer.beta
            
            # part use activation and bias
            act = np.dot(data, beta.T) + ae.layer.bias
            
            # if num_unit is equal, activation is linear
            #print data.shape[0], ae.n_hidden
            #if data.shape[0] != ae.n_hidden:
            data = self.activation(act)

            # append beta
            betas.append(beta)

        # set betas and data for fine_tune
        self.betas = betas
        self.data4fine = data
        return data

    def pre_extraction(self, input, limit=-1):
        # pre_extraction
        data = input

        if input == None:
            print 'input is none type.'
            return None
        else:
            for i, ae in enumerate(self.auto_encoders):
                beta = ae.layer.beta
                # beta = self.betas[i]
                #print "i:", i
                #print "data:", data
                #print "beta:", beta
                
                data = self.activation(np.dot(data, beta.T) + ae.layer.bias)
                if i == limit:
                    break
            
            return data



#################################################################
##  Stacked Extreme Learning Machine Correspondance AutoEncoders
#################################################################
   
class StackedELMCrossModalAutoEncoders(StackedELMAutoEncoder):
    """
    Stacked Extreme Learning Machine Correspondance AutoEncoders

    
    """
    
    def __init__(self, n_hidden=None, n_coef=None,
                 activation=sigmoid, seed=123):
        # initialize size of neuron
        if n_hidden is None:
            raise Exception("nlist_hidden is udefined")
        self.n_hidden = n_hidden
        if n_coef is None:
            n_coef = [None] * len(n_hidden)
        self.coef = n_coef
        self.activation = activation

        # initialize auto_encoder
        auto_encoders = []
        for i, num in enumerate(n_hidden):
            ae1 = ELMCrossModalAutoEncoder(activation=activation, n_hidden=num, coef=n_coef[i], seed=seed)
            ae2 = ELMCrossModalAutoEncoder(activation=activation, n_hidden=num, coef=n_coef[i], seed=seed)
            auto_encoders1.append(ae1)
            auto_encoders2.append(ae2)
        self.auto_encoders1 = auto_encoders1
        self.auto_encoders2 = auto_encoders2


    def pre_train(self, input1, input2):

        # pre_train
        print "pre_train"
        data1, data2 = input1, input2
        betas1, betas2 = [], []
        for i, (ae1, ae2) in enumerate(zip(self.auto_encoders1, self.auto_encoders2)):
            # fit auto_encoder
            print " ", i,"ae fit"
            ae1.fit(data1, data2)
            ae2.fit(data2, data1)

            # get beta
            beta1 = ae1.layer.beta
            beta2 = ae2.layer.beta
            
            # part use activation and bias
            act1 = np.dot(data1, beta.T) + ae1.layer.bias
            act2 = np.dot(data2, beta.T) + ae2.layer.bias
            
            # if num_unit is equal, activation is linear
            #print data.shape[0], ae.n_hidden
            #if data.shape[0] != ae.n_hidden:
            data1 = self.activation(act1)
            data2 = self.activation(act2)

            # append beta
            betas1.append(beta1)
            betas2.append(beta2)

        # set betas and data for fine_tune
        self.betas1 = betas1
        self.betas2 = betas2
        self.data4fine1 = data1
        self.data4fine2 = data2

        return data1, data2


    def pre_extraction(self, input1, input2, limit=-1):

        if input1:
            data1 = StackedELMAutoEncoder.pre_extraction(self, input1, limit=-1)
        else:
            data1 = None

        if input2:
            data2 = StackedELMAutoEncoder.pre_extraction(self, input2, limit=-1)
        else:
            data2 = None

        return data1, data2

    
class ELMAutoEncoder(object):
    """
    Extreme Learning Machine Auto Encoder
    
    
    """

    def __init__(self, activation=sigmoid,
                 n_hidden=50, coef=None, seed=128, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.n_hidden = n_hidden
        self.coef = coef
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain


    def construct(self, input):
        # set parameter of layer
        self.input = input
        self.n_input = len(input[0])
        self.n_output = len(input[0])
       
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # orthogonal weight and forcely regularization

        h, w = weight.shape
        if h < w:
            # print "height < width"
            weight = weight.T
            q, r = np.linalg.qr(weight[:h, :])
            weight[:h, :] = q.T
            weight = weight.T
        else:
            # print "height > width"
            weight[:w, :], r = np.linalg.qr(weight[:w, :])

        # print "weight_shape:"
        # print weight.shape
            
        # for i in xrange(np.min(weight.shape)):
        #     w = weight[i]
        #     for j in xrange(0,i):
        #         w = w - weight[j].dot(w) * weight[j]

        #     if np.linalg.norm(w) < 1:
        #         print i, np.linalg.norm(w), w.shape
            
        #     w = w / np.linalg.norm(w)
        #     weight[i] = w
            

        # bias regularization
        denom = np.linalg.norm(bias)
        if denom != 0:
            denom = bias / denom
        
        # set weight and bias
        self.weight = weight
        self.bias = bias     
        
            
        # initialize layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        
    def fit(self, input):
        # construct layer
        self.construct(input)

        # fit layer
        self.layer.fit(input, input)
        
    def fprop(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.fprop(i).tolist()
            predict_output.append(o)
        return predict_output
    
    def score(self, input, teacher):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.fprop(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length
    
    def error(self, input):
        # get error
        pre = self.fprop(input)
        err = pre - input
        err = err * err
        print "sum of err^2", err.sum()
        return err.sum()
    


class ELMCrossModalAutoEncoder(ELMAutoEncoder):
    """
    Extreme Learning Machine Correspondance Auto Encoder
    
    
    """

    def __init__(self, activation=sigmoid,
                 n_hidden=50, coef=None, seed=128, domain=[-1., 1.]):

        ELMAutoEncoder.__init__(self, activation=sigmoid, n_hidden=50, coef=None, seed=128, domain=[-1., 1.])


    def construct(self, input, output):
        # set parameter of layer
        self.input = input
        self.n_input = len(input[0])
        self.n_output = len(output[0])
       
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # orthogonal weight and forcely regularization

        h, w = weight.shape
        if h < w:
            # print "height < width"
            weight = weight.T
            q, r = np.linalg.qr(weight[:h, :])
            weight[:h, :] = q.T
            weight = weight.T
        else:
            # print "height > width"
            weight[:w, :], r = np.linalg.qr(weight[:w, :])

        # bias regularization
        denom = np.linalg.norm(bias)
        if denom != 0:
            denom = bias / denom
        
        # set weight and bias
        self.weight = weight
        self.bias = bias     
        
            
        # initialize layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        
    def fit(self, input, output):
        # construct layer
        self.construct(input, output)

        # fit layer
        self.layer.fit(input, output)



class ELMClassifier(object):
    """
    Extreme Learning Machine
    
    
    """

    def __init__(self, activation=sigmoid, vector='orthogonal',
                 coef=None, n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.vector = vector
        self.coef = coef
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta
    
    def construct(self, input, teacher):
        # set input, teacher and class
        self.input = input
        self.teacher = teacher
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)

        # weight and bias
        low, high = self.domain
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # condition : orthogonal random else
        if self.vector == 'orthogonal':
            print "set weight and bias orthogonaly"


            h, w = weight.shape
            if w > h:
                weight = weight.T
                q, r = np.linalg.qr(weight[:h, :])
                weight[:h, :] = q.T
            else:
                weight[:w, :], r = np.linalg.qr(weight[:w, :])


            # for i in xrange(np.min(weight.shape)):
            #     w = weight[i]
            #     for j in xrange(0,i):
            #         w = w - weight[j].dot(w) * weight[j]
            #     w = w / np.linalg.norm(w)
            #     weight[i] = w

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                denom = bias / denom
                    
        elif self.vector == 'random':
            print "set weight and bias randomly"
            # regularize weight
            
            #for i,w enumerate(weight.T):
            for i,w in enumerate(weight):
                denom = np.linalg.norm(w)
                if denom != 0:
                    #weight.T[i] = w / denom
                    weight[i] = w / denom

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                bias = bias / denom
                    
        else:
            print "warning: vector isn't orthogonal or random"
            
        
        # self weight and bias
        self.weight = weight
        self.bias = bias
            
        # self layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        
    def fit(self, input, teacher):
        # construct layer
        self.construct(input, teacher)

        # convert teacher to signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])

        # fit layer
        self.layer.fit(input, signal)
        
    def predict(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(o)
        #print "outputs", predict_output

        # get predict_classes from index of max_function(predict_output) 
        predict_classes = []
        for o in predict_output:
             predict_classes.append(self.classes[o.index(max(o))])
        #print "predict" predict_classes

        return predict_classes

    def score(self, input, label):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == label[i]: count += 1
        return count * 1.0 / length


if __name__ == "__main__":
    
    train = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    label = np.array([1, 1, -1, -1])
    test = np.array([[3, 3], [-3, -3]])

    model = MLELMClassifier(n_hidden=[4,8,5])

    model.fit(train, label)

    print model.predict(train)
    print model.predict(test)
    print "score:", model.score(train, label)
    """
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]
    
    model = ELMClassifier()
    model.fit(train, label)
    pre = model.predict(test)
    print pre
    print model.score(train, label)
    """
