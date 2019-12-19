import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

def getInput():
    n = int(input())
    X = []
    y = []
    for _ in range(n):
        obj = list(map(int, input().split()))
        y.append(obj[-1])
        X.append(obj[:-1])
    c = int(input())
    return X, y, c

def transpose(M):
    if not isinstance(M[0],list):
        M = [M]
 
    rows = len(M)
    cols = len(M[0])
 
    # Section 3: MT is zeros matrix with transposed dimensions
    MT = [[0 for _ in range(rows)] for _ in range(cols)]
 
    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]
    return MT
 

def dot(A, B):
    if not isinstance(A[0],list):
        A = [A]
    if not isinstance(B[0],list):
        B = [B]

    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if colsA != rowsB:
        print(A, B)
        print(colsA, rowsB)
        raise ArithmeticError(
            'Number of A columns must equal number of B rows.')
    C = [[0 for _ in range(colsB)] for _ in range(rowsA)]
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total
    return C

def zeros(n):
    return [0 for _ in range(n)]

def norm(X):
    return math.sqrt(sum([x**2 for x in X]))


class SVM():
    def __init__(self, max_iter=1000, kernel_type='linear', C=1.0, epsilon=0.00001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'gaussian' : self.kerel_gaussian,
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        n = len(X)
        self.alpha = zeros(n)
        self.b = 0
        self.kernel = self.kernels[self.kernel_type]
        self.X = X
        self.y = y
        epoch = 0

        while epoch < self.max_iter:
            # changed_alphas = 0
            alpha_prev = self.alpha.copy()

            for j in range(0, n):
                # pick two points
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i], X[j], y[i], y[j]

                # compute second derivative of an objective function
                k_ij = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2 * self.kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                
                # compute bounds for alpha_j
                alpha_i, alpha_j = self.alpha[i], self.alpha[j]
                (L, H) = self.compute_L_H(self.C, alpha_i, alpha_j, y_i, y_j)

                # compute error
                E_i = self.E(x_i, y_i)
                E_j = self.E(x_j, y_j)

                # set new alpha_j value
                self.alpha[j] = alpha_j + float(y_j * (E_i - E_j))/k_ij

                # clip alpha_j
                self.alpha[j] = max(self.alpha[j], L)
                self.alpha[j] = min(self.alpha[j], H)
                
                # set new alpha_i value
                self.alpha[i] = alpha_i + y_i*y_j * (alpha_j - self.alpha[j])

                # calculate b
                b1 = self.calc_b(x_i, x_j, E_i, (self.alpha[i], alpha_i), (self.alpha[j], alpha_j), y_i, y_j, 1)
                b2 = self.calc_b(x_i, x_j, E_i, (self.alpha[i], alpha_i), (self.alpha[j], alpha_j), y_i, y_j, 2)

                # clip b
                if 0 <= self.alpha[i] <= self.C:
                    self.b = b1
                elif 0 <= self.alpha[j] <= self.C:
                    self.b = b2
                else:
                    self.b = (b1+b2)/2 

            # check convergence
            diff = norm([a - a_prev for a, a_prev in zip(self.alpha, alpha_prev)])
            # if diff < self.epsilon:
            #     break
            epoch += 1

            # if count >= self.max_iter:
                # print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                # return

        # Compute final model parameters
        # self.b = self.calc_b(X, y, self.w)
        # if self.kernel_type == 'linear':
        #     self.w = self.calc_w(alpha, y, X)
        
        # alpha_idx = np.where(alpha > 0)[0]
        # support_vectors = X[alpha_idx,:]
        print(epoch)
        return self.alpha, self.b

    def predict(self, u):
        return self.h(u)

    def calc_b(self, x_i, x_j, E, a_i, a_j, y_i, y_j, x_type):
        if x_type == 1:
            return self.b - E - y_i*(a_i[0] - a_i[1])*self.kernel(x_i, x_i) - y_j*(a_j[0] - a_j[1])*self.kernel(x_i, x_j)
        else:
            return self.b - E - y_i*(a_i[0] - a_i[1])*self.kernel(x_i, x_j) - y_j*(a_j[0] - a_j[1])*self.kernel(x_j, x_j)

    # prediction
    def h(self, u):
        s = 0
        for i in range(len(X)):
            s+=self.alpha[i]*self.y[i]*self.kernel(self.X[i], u)
        s/=len(X)
        return 1 if (s+self.b)>=0 else -1

    # prediction error
    def E(self, x_k, y_k):
        return self.h(x_k) - y_k

    # alpha bounds
    def compute_L_H(self, C, alpha_i, alpha_j, y_i, y_j):
        if(y_i != y_j):
            return (max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j))
        else:
            return (max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j))

    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i

    # kernel functions
    def kernel_linear(self, x1, x2):
        return dot(x1, transpose(x2))[0][0] + 1

    def kernel_quadratic(self, x1, x2):
        return ((dot(x1, transpose(x2))[0][0] + 1) ** 2)

    def kerel_gaussian(self, x1, x2):
        gamma = 1.
        return math.exp(-1*norm([x1_ - x2_ for x1_, x2_ in zip(x1, x2)])/gamma)
        
from sklearn import svm
model = svm.SVC()
X, y, c = getInput()
# model = SVM(C=c, kernel_type="gaussian")
model.fit(X, y)
print(model.classes_)