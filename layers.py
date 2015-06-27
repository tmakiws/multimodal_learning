import numpy as np
from mycca import MyCCA as CCA
from scipy.linalg import eig

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2


class Layer(object):
    def __init__(self, activation, size, weight, bias, c=None, name=None):
        # initialize
        self.name = name
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.c = c
        self.w = weight
        self.bias = bias
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])

    #def get_beta(self):
    #    return self.beta
        
    def get_i2h(self, input):
        # print self.w.T.shape, input.shape, self.bias.shape
        return self.activation(np.dot(self.w.T, input) + self.bias)

    def get_h2o(self, hidden):
        return np.dot(self.beta.T, hidden)

    def fprop(self, input):
        hidden = self.get_i2h(input)  # from input to hidden
        output = self.get_h2o(hidden) # from hidden to output
        return output
    
    def fit(self, input, signal):
        """
        fit : set beta
        risk : memory error (dot)
        """
        
        if self.name:
            print 'Layer: ', self.name
        # get activation of hidden layer
        H = []
        for i, d in enumerate(input):
            if i % 100 == 99:
                pass
                #0427
                #sys.stdout.write("\r    input %d" % (i+1))
                #sys.stdout.flush()
            H.append(self.get_i2h(d))
        print " done."

        print ' num_unit: ', input.shape[1], '=>', len(H[0])
        # coefficient of regularization
        #sys.stdout.write("\r    coefficient")
        #sys.stdout.flush()
        H = np.array(H)
        np_id = np.identity(min(np.array(H).shape))

        if H.shape[0] < H.shape[1]:
            sigma = np.dot(H, H.T)
        else:
            sigma = np.dot(H.T, H)

        #if self.c == None:
        if self.c != 0 and self.c:
            print ' coefficient', self.c * np.average(np.diag(sigma))
            regular = self.c * np.average(np.diag(sigma)) * np.eye(sigma.shape[0])
            #print 'new regularizer'
        else:
            regular = 0
        #else:
        #    coefficient = 1. / self.c
        #    regular = coefficient * np_id
        print " done."

        # pseudo inverse
        #sys.stdout.write("\r    pseudo inverse")
        #sys.stdout.flush()
        # print H.shape, H

        #Hp = np.linalg.pinv(H)
        Hp = np.linalg.inv(sigma + regular)
        if H.shape[0] < H.shape[1]:
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.dot(Hp, H.T)
        print " done."
            
        # set beta
        #sys.stdout.write("\r    set beta")
        #sys.stdout.flush()
        self.beta = np.dot(Hp, np.array(signal))
        print " done."


class CCALayer(object):

    def __init__(self, activation, n_comp, cca_param, joint=False):
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.cca = CCA(n_components=n_comp, reg_param=cca_param, show_runtime=True)
        self.joint = joint

    def fit(self, input1, input2):

        self.cca.fit(input1, input2)
        #self.beta1 = self.cca.x_weights
        #self.beta2 = self.cca.y_weights

        if self.joint:
            # PCCA
            out = self.cca.ptransform(input1, input2)
            return self.activation(out)

        else:
            # CCA
            out1, out2 = self.cca.transform(input1, input2)
            return self.activation(out1), self.activation(out2)


    def fprop(self, input1, input2):

        if self.joint:
            out = self.activation(self.cca.ptransform(input1, input2))
            return out

        else:
            out1, out2 = self.activation(self.cca.transform(input1, input2))
            return out1, out2

        # if input2:
        #     if self.joint:
        #         out2 = self.activation(self.cca.ptransform(input2))
        #     else:
        #         out2 = self.activation(self.cca.transform(input2))
        # else:
        #     out2 = None



class semiCCALayer(CCALayer):

    def __init__(self, activation, n_comp, cca_param):
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.cca = semiCCA(n_components=n_comp, reg_param=cca_param, beta=0.5, show_runtime=True)
        self.joint = False


class HeMapLayer(Layer):
    
    def __init__(self, n_hidden, weights=None, lamb=None, activation=sigmoid, name=None):

        self.name = name
        # self.activation = activation
        self.n_hidden = n_hidden
        self.lamb = lamb
        self.activation = activation

        if weights != None:
            self.weight1, self.weight2 = weights
        else:
            self.weight1, self.weight2 = [None, None]
     
       
    def fprop(self, input1, input2):
        
        if input1 != None:
            out1 = self.activation(np.dot(input1, np.linalg.pinv(self.weight1)))
        else:
            print 'out1 is missing.'
            out1 = None

        if input2 != None:
            out2 = self.activation(np.dot(input2, np.linalg.pinv(self.weight2)))
        else:
            print 'out2 is missing.'
            out2 = None

        return out1, out2
 
    
    def fit(self, input1, input2):
        """
        fit : set beta
        risk : memory error (dot)
        """
        #print self.lamb
        if self.name:
            print 'Layer: ', self.name
        
        S_ST = np.dot(input1, input1.T)
        T_TT = np.dot(input2, input2.T)
        A1 = 2*T_TT + (self.lamb**2/2.)*S_ST
        A2 = self.lamb*(S_ST + T_TT)
        A4 = 2*S_ST + (self.lamb**2/2.)*T_TT
        
        print A1.shape, A2.shape, A4.shape
        A = np.vstack((np.hstack((A1, A2)), np.hstack((A2.T, A4))))
        
        eig_vals, eig_vecs = eig(A)

        eig_dim = np.linalg.matrix_rank(A)
        n_out = min(self.n_hidden, eig_dim)
        #np.save("matrixA.npy", A)
        print 'rank:', eig_dim, 'n_hidden: ', self.n_hidden
        print ' num_unit: (%s, %s) => (%s, %s)' % (input1.shape[1], input2.shape[1], n_out, n_out)
        print 'eig_vals: ', eig_vals[:n_out].astype('double')
        l = len(eig_vecs)
        
        # Q: arbitrary orthogonal matrix
        # Q = np.eyes(1600)

        projected1 = eig_vecs.T[:l/2,:n_out].astype('double')
        projected2 = eig_vecs.T[l/2:,:n_out].astype('double')
        
        print 'n_unit: ', projected1.shape
        self.weight1 = 2*np.dot(projected1.T, input1) + self.lamb*np.dot(projected2.T, input1) / (2 + self.lamb)
        self.weight2 = 2*np.dot(projected2.T, input2) + self.lamb*np.dot(projected1.T, input2) / (2 + self.lamb)
        
        return projected1, projected2
        


if __name__ == "__main__":
    
    train = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    label = np.array([1, 1, -1, -1])
    test = np.array([1.5, 1.5])

    print 'LayerTest'
    layer = Layer(sigmoid,
                  [2, 5, 1],
                  np.random.random((2,3)),
                  np.random.random((3)),
                  c=1,name='test')
    layer.fit(train, label)
    print layer.fprop(test)

   # train2 = np.array([[2,0], []])
   # print 'HeMapTest'
   # hmlayer = HeMapLayer(sigmoid,
   #                      [2,])
