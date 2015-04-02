#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eig
import time

# from mypool import MyPool

class MyCCA(object):

    def __init__(self, n_components=2, reg_param=0.1, calc_time=False):

        self.n_components = n_components
        self.reg_param = reg_param
        self.x_weights = None
        self.x_eigvals = None
        self.y_weights = None
        self.y_eigvals = None
        self.x_mean = None
        self.y_mean = None
        self.calc_time = calc_time

               
    def get_params(self):
        print "===================="
        print "  CCA parameters    "
        print "===================="
        print " | "
        print " |- n_components: %s" % self.n_components
        print " |- reg_param:    %s" % self.reg_param
        print " |- calc_time:    %s" % self.calc_time

        

        
    def solve_eigprob(self, left, right):
        
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])
        # print eig_dim
        # print "solve_eigprob"
        
        eig_vals, eig_vecs = eig(left, right)# ;print eig_vals.imag
        
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        
        
        # regularization
        # eig_vecs = np.dot(eig_vecs, np.diag(np.reciprocal(np.linalg.norm(eig_vecs, axis=0))))
        # eig_vecs = np.dot(eig_vecs, np.diag(np.reciprocal(np.linalg.norm(eig_vecs, axis=0))))
        print right.shape, eig_vecs.shape
        
        var = np.dot(eig_vecs.T, np.dot(right, eig_vecs))
        # print var
        invvar = np.diag(np.reciprocal(np.sqrt(np.diag(var))))
        # print invvar
        eig_vecs = np.dot(eig_vecs, invvar)
        
        print np.dot(eig_vecs.T, np.dot(right, eig_vecs))

        return eig_vals[:self.n_components], eig_vecs[:,:self.n_components]

    
    def fit(self, xs, ys):

        #-------- 平均を引く
        xm = np.mean(xs, axis=0)
        self.x_mean = xm
        xs = xs - xm
        
        ym = np.mean(ys, axis=0)
        self.y_mean = ym
        ys = ys - ym
        
        #-------- CCA の平均分散共分散を計算.
        zs = np.vstack((xs.T, ys.T))
        # print "Cov(zs)"
        Cov = np.cov(zs)
        # print "len(xs.T)"
        p = len(xs.T)
        # print "Cxx, Cyy, Cxy"
        Cxx = Cov[:p, :p]
        Cyy = Cov[p:, p:]
        Cxy = Cov[:p, p:]
        
        # print Cxx.shape, Cxy.shape, Cyy.shape
        #--------

        
        #-------- CCA の一般化固有値問題 ( A*u = (lambda)*B*u ).

        start = time.time()
        
        # 正則化項を加える.
        Cxx += self.reg_param * np.average(np.diag(Cxx)) * np.eye(Cxx.shape[0])
        Cyy += self.reg_param * np.average(np.diag(Cyy)) * np.eye(Cyy.shape[0])
        
        # left = A, right = B
        xleft = np.dot(Cxy, np.linalg.solve(Cyy,Cxy.T))
        xright = Cxx
        x_eigvals, x_eigvecs = self.solve_eigprob(xleft, xright)
        
        yleft = np.dot(Cxy.T, np.linalg.solve(Cxx,Cxy))
        yright = Cyy
        y_eigvals, y_eigvecs = self.solve_eigprob(yleft, yright)  

        # d = min(len(x_eigvals), len(y_eigvals))
        # print d
        self.x_weights = x_eigvecs
        self.y_weights = y_eigvecs
        self.eigvals = x_eigvals
        

        if self.calc_time:
            print "Fitting done in %.2f sec." % (time.time() - start)
        
        # print x_eigvals.shape, y_eigvecs.shape
        # print x_eigvals
        #--------
        

    def x_transform(self, x):
        # x -= self.x_mean 
        x_projected = np.dot(x, self.x_weights[:,:self.n_components])
        return x_projected

    def x_ptransform(self, x):
        # x -= self.x_mean
        lamb = np.diag(self.eigvals)
        z = np.dot(np.dot(x, self.x_weights[:,:self.n_components]), lamb**0.5)
        return z

    def y_transform(self, y):
        # y -= self.y_mean
        y_projected = np.dot(y, self.y_weights[:,:self.n_components])
        return y_projected

    def y_ptransform(self, y):
        # y -= self.y_mean
        lamb = np.diag(self.eigvals)
        z = np.dot(np.dot(y, self.y_weights[:,:self.n_components]), lamb**0.5)
        return z

    # def transform_y2x(self, y_c):
    #     W = np.dot(self.xs, np.linalg.solve(self.Cxx, self.Cxy))
    #     print W.shape
    #     y_projected = np.dot(np.diag(self.x_eigvals), np.dot(W, y_c))
    #     return y_projected
        
    def transform(self, x, y):

        #-------- CCA で射影.
        start = time.time()
        # print x.shape, x_eigvecs[:,:dim].shape
        # print y.shape, y_eigvecs[:,:dim].shape
        # x -= self.x_mean
        # y -= self.y_mean
        
        x_projected = np.dot(x, self.x_weights)#[:,:self.n_components])
        y_projected = np.dot(y, self.y_weights)#[:,:self.n_components])

        if self.calc_time:
            print "Transforming done in %.2f sec." % (time.time() - start)

        return x_projected, y_projected
        #--------

    def ptransform(self, x, y, beta=0.5):

        start = time.time()
        print x.shape
        # x -= self.x_mean
        # y -= self.y_mean

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
        q = np.vstack((x_projected.T, y_projected.T))
        # print p.T.shape, mat.shape, q.shape
        z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]

        if self.calc_time:
            print "Transforming done in %.2f sec." % (time.time() - start)

        return z

    
    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

    def fit_ptransform(self, x, y, beta=0.5):
        self.fit(x, y)
        return self.ptransform(x, y, beta)



if __name__=="__main__":
    
    # Reduce dimensions of x, y from 30, 20 to 10, 10 respectively.
    x = np.random.random((100, 30))
    y = np.random.random((100, 20))
    cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)
    x_c, y_c = cca.fit_transform(x, y)
        
    # 
    print np.corrcoef(x_c[:,0], y_c[:,0])
