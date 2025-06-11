import numpy as np
import mpmath as mm
import scipy


class MathModule:

    def __init__(self, high_precision):
        self.high_precision = high_precision


    def exp(self, a):
        if self.high_precision:
            return mm.exp(a)
        else:
            return np.exp(a)

    def sqrt(self,a):
        if self.high_precision:
            return mm.sqrt(a)
        else:
            return np.sqrt(a)

    def matmul(self,a, b):
        if self.high_precision:
            return a*b
        else:
            return np.matmul(a,b)

    def erf(self,a):
        if self.high_precision:
            return mm.erf(a)
        else:
            return scipy.special.erf(a)

    def norm(self,a, ord=2):
        if self.high_precision:
            return mmm.norm(a, p=ord)
        try:
            return np.linalg.norm(a,ord=ord)
        except: #numpy doesn't like you to pass ord when a is scalar.
            return np.linalg.norm(a)

    def ones(self,N):
        if self.high_precision:
            return mm.matrix(np.ones(N))
        else:
            return np.ones(N)

    def inv(self,a):
        if self.high_precision:
            # Precision is already set by GP_recipe class if high_precision=True
            try:
                return a**-1
            except:
                print(f"Error inverting matrix: {a}")
                print("Returning zeros matrix instead.")
                return self.zeros_matrix(a.rows, a.cols)
        else:
            # Removed quadruple_precision logic. Use high_precision=True in GP_recipe for higher precision inversion.
            return np.linalg.inv(a)
    
    def pinv(self, a):
        if self.high_precision:
            return a.pinv()
        else:
            return np.linalg.pinv(a)

    def cond(self,A, p=None):
        if self.high_precision:
            try:
                return mm.cond(A)
            except:
                print(f"Error cond")
                return -1
        else:
            return np.linalg.cond(A, p=p)

    def zeros_matrix(self,Ny,Nx):
        '''helper function to return matrix'''
        ma = np.zeros((Ny,Nx))

        if self.high_precision:
            ma = mm.matrix(ma.tolist())

        if ((Ny==1) and (not self.high_precision)):
            ma = ma[0] #convert to vector

        return ma

    def mean(self, arr):
        '''Returns mean of arr'''

        if self.high_precision:
            return mm.fsum(arr)/len(arr)
        else:
            return np.mean(arr)