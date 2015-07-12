#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eig
import time
import matplotlib.pyplot as plt
import sys
from math import pi, sqrt, exp  

class MyCCA(object):

    def __init__(self, n_components=None, reg_param=0.1, show_runtime=False, metric='euc'):

        self.n_components = n_components
        self.reg_param = reg_param
        self.x_weights = None
        self.eigvals = None
        self.y_weights = None
        self.x_mean = None
        self.y_mean = None
        self.Cxx = None
        self.Cyy = None
        self.Cxy = None
        self.Cyx = None
        self.show_runtime = show_runtime
        self.metric = metric

               
    def get_params(self):
        print "===================="
        print "  CCA parameters    "
        print "===================="
        print " | "
        print " |- n_components: %s" % self.n_components
        print " |- reg_param:    %s" % self.reg_param
        print " |- show_runtime:    %s" % self.show_runtime

        
    def solve_eigprob(self, left, right):
        
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])
        print 'rank:', eig_dim, '(left: %s, right: %s)' % (np.linalg.matrix_rank(left), np.linalg.matrix_rank(right))

        if self.n_components == None:
            self.n_components = eig_dim
        # print eig_dim
        # print "solve_eigprob"
        
        eig_vals, eig_vecs = eig(left, right)# ;print eig_vals.imag
        
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        # regularization
        # eig_vecs = np.dot(eig_vecs, np.diag(np.reciprocal(np.linalg.norm(eig_vecs, axis=0))))
        #print right.shape, eig_vecs.shape
        
        var = np.dot(eig_vecs.T, np.dot(right, eig_vecs))
        # print var
        invvar = np.diag(np.reciprocal(np.sqrt(np.diag(var))))
        # print invvar
        eig_vecs = np.dot(eig_vecs, invvar)
        
        #print "U.T*C*U"
        #print np.dot(eig_vecs.T, np.dot(right, eig_vecs))

        return eig_vals[:self.n_components], eig_vecs[:,:self.n_components]

    
    def compute_mean_and_covariance(self, xs, ys):

        #-------- 平均を引く
        xm = np.mean(xs, axis=0)
        self.x_mean = xm
        x = xs - xm
        
        ym = np.mean(ys, axis=0)
        self.y_mean = ym
        y = ys - ym


        #-------- CCA の平均分散共分散を計算.
        z = np.vstack((x.T, y.T))
        # print "Cov(z)"
        Cov = np.cov(z)
        # print "len(x.T)"
        p = len(x.T)
        # print "Cxx, Cyy, Cxy"
        self.Cxx = Cov[:p, :p]
        self.Cyy = Cov[p:, p:]
        self.Cxy = Cov[:p, p:]
        self.Cyx = self.Cxy.T
        
        #print self.Cxx.shape, self.Cxy.shape, self.Cyy.shape


    def fit(self):
        #-------- CCA の一般化固有値問題 ( A*u = (lambda)*B*u ).

        start = time.time()
        
        print self.Cxx.shape
        # 正則化項を加える.
        self.Cxx += self.reg_param * np.average(np.diag(self.Cxx)) * np.eye(self.Cxx.shape[0])
        self.Cyy += self.reg_param * np.average(np.diag(self.Cyy)) * np.eye(self.Cyy.shape[0])
     
        # left = A, right = B
        xleft_right = np.linalg.solve(self.Cyy, self.Cyx)
        xleft = np.dot(self.Cxy, xleft_right)
        xright = self.Cxx
        self.eigvals, self.x_weights = self.solve_eigprob(xleft, xright)
        
        #print np.linalg.matrix_rank(self.Cyy), np.linalg.matrix_rank(self.Cxy), 

        # B = inv(Cyy) * Cyx * A * inv(diag(eigvals))
        # print xleft_right.shape, self.x_weights.shape, np.diag(np.reciprocal(self.eigvals))
        self.y_weights = np.dot(xleft_right, np.dot(self.x_weights, np.diag(np.reciprocal(self.eigvals))))
        #print self.y_weights
        

        #yleft = np.dot(self.Cyx, np.linalg.solve(self.Cxx, self.Cxy))
        #yright = self.Cyy
        #y_eigvals, y_eigvecs = self.solve_eigprob(yleft, yright)  
        #print y_eigvecs

        if self.metric == 'cos':
            self.x_weights = np.dot(self.x_weights, np.diag(self.eigvals*self.eigvals*self.eigvals*self.eigvals))
            self.y_weights = np.dot(self.y_weights, np.diag(self.eigvals*self.eigvals*self.eigvals*self.eigvals))
        
        if self.show_runtime:
            print "Fitting done in %.2f sec." % (time.time() - start)
        
        # print x_eigvals.shape, y_eigvecs.shape
        # print x_eigvals
        #--------
        

    def x_transform(self, xs):
        x = xs - self.x_mean 
        x_projected = np.dot(x, self.x_weights[:,:self.n_components])
        return x_projected

    def x_ptransform(self, xs):
        x = xs - self.x_mean
        lamb = np.diag(self.eigvals)
        z = np.dot(np.dot(x, self.x_weights[:,:self.n_components]), lamb**0.5)
        return z

    def y_transform(self, ys):
        y = ys - self.y_mean
        y_projected = np.dot(y, self.y_weights[:,:self.n_components])
        return y_projected

    def y_ptransform(self, ys):
        y = ys - self.y_mean
        lamb = np.diag(self.eigvals)
        z = np.dot(np.dot(y, self.y_weights[:,:self.n_components]), lamb**0.5)
        return z

    # def transform_y2x(self, y_c):
    #     W = np.dot(self.xs, np.linalg.solve(self.Cxx, self.Cxy))
    #     print W.shape
    #     y_projected = np.dot(np.diag(self.x_eigvals), np.dot(W, y_c))
    #     return y_projected
        
    def transform(self, xs, ys):

        if xs != None and ys != None:

            #-------- CCA で射影.
            start = time.time()
            # print xs.shape, x_eigvecs[:,:dim].shape
            # print ys.shape, y_eigvecs[:,:dim].shape
            #print 'trans xs, xsmean', xs, self.x_mean
            x = xs - self.x_mean
            y = ys - self.y_mean
            #print np.mean(x, axis=0)#.shape, self.x_weights.shape
            x_projected = np.dot(x, self.x_weights)#[:,:self.n_components])
            #print np.mean(y, axis=0), self.y_weights.shape
            y_projected = np.dot(y, self.y_weights)#[:,:self.n_components])
            
            # 負の相関の場合の処理
            #for i in xrange(self.n_components):
            #    # print np.corrcoef(x_projected[:,i], y_projected[:,i])
            #    if np.corrcoef(x_projected[:,i], y_projected[:,i])[0,1] < 0:
            #        y_projected[:,i] *= -1
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)
            

        elif xs != None:
            start = time.time()
            x_projected = self.x_transform(xs)
            y_projected = None
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)


        elif ys != None:
            start = time.time()
            x_projected = None
            y_projected = self.y_transform(ys)
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)


        return x_projected, y_projected
        #--------

    def ptransform(self, xs, ys, beta=0.5):

        if xs != None and ys != None:
            start = time.time()
            #print xs.shape
            x = xs - self.x_mean
            y = ys - self.y_mean
            
            # print x.shape, self.x_weights_.shape
            x_projected = np.dot(x, self.x_weights)
            y_projected = np.dot(y, self.y_weights)
            
            I = np.eye(len(self.eigvals))
            lamb = np.diag(self.eigvals)
            mat1 = np.linalg.solve(I - np.diag(self.eigvals**2), I)
            mat2 = -np.dot(mat1, lamb)
            mat12 = np.vstack((mat1, mat2))
            mat21 = np.vstack((mat2, mat1))
            mat = np.hstack((mat12, mat21))
            # print lamb.shape, lamb
            p = np.vstack((lamb**beta, lamb**(1-beta)))
            #print x_projected.shape, y_projected.shape
            q = np.vstack((x_projected.T, y_projected.T))
            # print p.T.shape, mat.shape, q.shape
            z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]
            
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)

        elif xs != None:
            start = time.time()
            z = self.x_ptransform(xs)
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)


        elif ys != None:
            start = time.time()
            z = self.y_ptransform(ys)
            if self.show_runtime:
                print "Transforming done in %.2f sec." % (time.time() - start)

        return z

    
    def fit_transform(self, x, y):
        self.compute_mean_and_covariance(x, y)
        self.fit()
        return self.transform(x, y)


    def fit_ptransform(self, x, y, beta=0.5):
        self.compute_mean_and_covariance(x, y)
        self.fit()
        return self.ptransform(x, y, beta)


class semiCCA(MyCCA):
    
   def __init__(self, n_components=None, reg_param=0.1, beta=0.5, show_runtime=False):
       MyCCA.__init__(self, n_components=n_components, reg_param=reg_param, show_runtime=show_runtime)
       self.beta = beta

   def fit(self):
        #-------- CCA の一般化固有値問題 ( A*u = (lambda)*B*u ).

        start = time.time()
        
        # 正則化項を加える.
        self.Cxx += self.reg_param * np.average(np.diag(self.Cxx)) * np.eye(self.Cxx.shape[0])
        self.Cyy += self.reg_param * np.average(np.diag(self.Cyy)) * np.eye(self.Cyy.shape[0])

        n_pairs = self.Cxx.shape[0]
        #l = 1
        #self.Cxx += l*
        #self.Cyy += l*

        
        # left = A, right = B
        xleft = np.vstack(( np.hstack(( (1-self.beta)*self.Cxx, self.beta*self.Cxy )), np.hstack(( self.beta*self.Cxy.T, (1-self.beta)*self.Cyy )) ))

        xright = np.zeros(( xleft.shape[0], xleft.shape[0] ))
        xright[:n_pairs, :n_pairs] = self.Cxx
        xright[n_pairs:, n_pairs:] = self.Cyy
        xright = self.beta*xright + (1-self.beta)*np.eye(xleft.shape[0])


        self.eigvals, weights = self.solve_eigprob(xleft, xright)
        
        print np.linalg.matrix_rank(self.Cyy), np.linalg.matrix_rank(self.Cxy), 

        self.x_weights = weights[:n_pairs]
        self.y_weights = weights[n_pairs:]
        print self.x_weights.shape, self.y_weights.shape

        # d = min(len(x_eigvals), len(y_eigvals))
        # print d

        if self.show_runtime:
            print "Fitting done in %.2f sec." % (time.time() - start)
        
        #--------



def cca_test(n_components=30):
    # Reduce dimensions of x, y from 30, 20 to 10, 10 respectively.
    tra_x = np.random.random((1000, 200))+100
    #print 'initial', tra_x
    tra_y = np.random.random((1000, 200))
    cca = MyCCA(n_components=n_components, reg_param=0.01, show_runtime=True)
    tra_x_c, tra_y_c = cca.fit_transform(tra_x, tra_y)
    #print 'after fit_trans', tra_x
    plt.plot(tra_x_c[:,0], tra_x_c[:,1], "r-")
    plt.plot(tra_y_c[:,0], tra_y_c[:,1], "b-")
    plt.show()
    corr = []
    for i in xrange(n_components):
        corr.append(np.corrcoef(tra_x_c[:,i], tra_y_c[:,i])[0,1])
    np.set_printoptions(precision=3)
    print np.array(corr)
    print np.array([ sqrt(e) for e in cca.eigvals ])

    #print 'before test', tra_x
    tes_x = tra_x + np.random.normal(0,0.2,size=(1000, 200))
    #print 'tes_x', tes_x
    tes_y = tra_y + np.random.normal(0,0.2,size=(1000, 200))

    tes_x_c, tes_y_c = cca.transform(tes_x, tes_y)
    plt.plot(tes_x_c[:,0], tes_x_c[:,1], "r-")
    plt.plot(tes_y_c[:,0], tes_y_c[:,1], "b-")
    plt.show()
    corr = []
    for i in xrange(n_components):
        corr.append(np.corrcoef(tes_x_c[:,i], tes_y_c[:,i])[0,1])
    np.set_printoptions(precision=3)
    print np.array(corr)
    #print np.array([ sqrt(e) for e in cca.eigvals ])

    
def pcca_test():
    # Reduce dimensions of x, y from 30, 20 to 10, 10 respectively.
    x = np.random.random((1000, 200))+100
    y = np.random.random((1000, 200))
    cca = MyCCA(n_components=10, reg_param=0.1, show_runtime=True)
    z_c = cca.fit_ptransform(x, y)
    plt.plot(z_c[:,0], z_c[:,1], "ro")
    print [ sqrt(e) for e in cca.eigvals ]

    tes_x = x + np.random.normal(0,0.3,size=(1000, 200))
    tes_y = y + np.random.normal(0,0.3,size=(1000, 200))

    tes_z_c = cca.ptransform(tes_x, tes_y)
    plt.plot(tes_z_c[:,0], tes_z_c[:,1], "bo")
    plt.show()
    #corr = []
    
    #for i in xrange(n_components):
    #    corr.append(np.corrcoef(tes_x_c[:,i], tes_y_c[:,i])[0,1])
    #np.set_printoptions(precision=3)
    #print np.array(corr)


def semicca_test(n_components=30):
    # Reduce dimensions of x, y from 30, 20 to 10, 10 respectively.
    tra_x = np.random.random((1000, 200))+100
    #print 'initial', tra_x
    tra_y = np.random.random((1000, 200))
    cca = semiCCA(n_components=n_components, reg_param=0.01, show_runtime=True)
    tra_x_c, tra_y_c = cca.fit_transform(tra_x, tra_y)
    #print 'after fit_trans', tra_x
    plt.plot(tra_x_c[:,0], tra_x_c[:,1], "r-")
    plt.plot(tra_y_c[:,0], tra_y_c[:,1], "b-")
    plt.show()
    corr = []
    for i in xrange(n_components):
        corr.append(np.corrcoef(tra_x_c[:,i], tra_y_c[:,i])[0,1])
    np.set_printoptions(precision=3)
    print np.array(corr)
    print np.array([ sqrt(e) for e in cca.eigvals ])

    #print 'before test', tra_x
    tes_x = tra_x + np.random.normal(0,0.2,size=(1000, 200))
    #print 'tes_x', tes_x
    tes_y = tra_y + np.random.normal(0,0.2,size=(1000, 200))

    tes_x_c, tes_y_c = cca.transform(tes_x, tes_y)
    plt.plot(tes_x_c[:,0], tes_x_c[:,1], "r-")
    plt.plot(tes_y_c[:,0], tes_y_c[:,1], "b-")
    plt.show()
    corr = []
    for i in xrange(n_components):
        corr.append(np.corrcoef(tes_x_c[:,i], tes_y_c[:,i])[0,1])
    np.set_printoptions(precision=3)
    print np.array(corr)
    #print np.array([ sqrt(e) for e in cca.eigvals ])



if __name__=="__main__":
    
    argvs = sys.argv
    
    if argvs[1] == 'cca':
       cca_test()
    elif argvs[1] == 'pcca':
       pcca_test()
    elif argvs[1] == 'semicca':
       semicca_test()
