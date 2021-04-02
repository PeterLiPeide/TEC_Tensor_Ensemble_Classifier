"""
CP-Support Tensor Machine (CP-STM) Solver

This moduler provides various functions to estimate a support tensor machine for CP tensors

dependencies: numpy, CVXOPT, tensorly

Author: Peide (Peter) Li

"""
#%%
from os import truncate
import numpy as np 
from cvxopt import matrix, solvers
import tensorly as tl 
from tensorly.decomposition import parafac
import time
from classification_tools import classification_metrics

"""Basic Function for SVM Solving"""
def SVM_QP(K, y, C = None, **kwargs):
    """Solving SVM from Dual / feature space using quatradic programming
        return the lagrange multipliers
        Input: K: kernel matrix (if not kernel, K becomes the outer product of x and x)
               y: labels
               C: slake parameter (for soft margin)
        truncate (optional): threshold parameter for supprt vector, i.e. if lagrange multiplier is greater than the threshold, the value is retained

        Output length = length y with non-SV value set to be zero

        (offset b is not sure; Others are double checked and resutls are identical to sklearn package)
    """
    if "truncate" not in kwargs:
        truncate = 1e-4
    else:
        truncate = kwargs['truncate']

    y = np.array(y)
    n_sample = y.shape[0]
    P = matrix(np.outer(y, y) * K, tc = 'd')
    q = matrix(np.ones(n_sample) * -1, tc = 'd')
    A = matrix(y, (1, n_sample), tc = 'd')
    b = matrix([0], tc = 'd')
    if not C:
        G = matrix(np.eye(n_sample) * -1)
        h = matrix(np.zeros(n_sample))
    else:
        tmp1 = np.eye(n_sample) * -1
        tmp2 = np.eye(n_sample)
        G = matrix(np.vstack((tmp1, tmp2)), tc = 'd')  #row wise stack
        tmp1 = np.zeros(n_sample)
        tmp2 = np.ones(n_sample) * C
        h = matrix(np.hstack((tmp1, tmp2)), tc = 'd') #column wise stack
    solution = solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(solution['x'])
    idx = np.where(alpha <= truncate)
    alpha[idx] = 0
    
    b = 0
    count_sv = 0
    for i in range(alpha.shape[0]):
        if alpha[i] > 0:
            count_sv += 1
            b += y[i] - np.sum(alpha * y * K[:, i])
    b /= count_sv

    return alpha, b, count_sv



def SVM_SQHinge(K, y, lamb, **kwargs):
    """
    Solving SVM using Squared Hinge Loss and Gaussian-Newton Methods
    eturn the lagrange multipliers
        Input: K: kernel matrix (if not kernel, K becomes the outer product of x and x)
               y: labels
               lamb: tuninig parameter
        truncate (optional): threshold parameter for supprt vector, i.e. if lagrange multiplier is greater than the threshold, the value is retained

        Output length = length y with non-SV value set to be zero

    """
    if 'truncate' not in kwargs:
        truncate = 1e-4
    else:
        truncate = kwargs['truncate']
    
    if 'maxiter' not in kwargs:
        maxiter = 30
    else:
        maxiter = kwargs['maxiter']
    if 'eta' not in kwargs:
        eta = 1e-3
    else:
        eta = kwargs['eta']

    y = np.array(y)
    n_sample = y.shape[0]
    current_iter = 0
    alpha_new = np.ones(n_sample)
    alpha_current = np.ones(n_sample)
    current_eta = 1000
    Dy = np.diag(y)
    while current_eta > eta and current_iter <= maxiter:
        alpha_current = alpha_new[:]
        tmp = np.dot(Dy, alpha_current)
        yc = np.dot(K, tmp)
        sv_ind = [1 if yc[i] < 1 else 0 for i in range(n_sample)]
        Is = np.diag(sv_ind)
        inv_mm = np.linalg.pinv(lamb * np.eye(n_sample) + (1 / n_sample) * np.dot(Is, K))
        lf_mm = (1 / n_sample) * np.dot(Dy, inv_mm)
        rs_mm = np.dot(Is, y)
        alpha_new = np.dot(lf_mm, rs_mm)
        current_eta = np.linalg.norm(alpha_new - alpha_current)
        current_iter += 1
    idx = np.where(alpha_new < truncate)
    alpha_new[idx] = 0

    b = 0
    count_sv = 0
    for i in range(n_sample):
        if alpha_new[i] > 0:
            count_sv += 1
            b += y[i] - np.sum(alpha_new * y * K[:, i])
    b /= count_sv
    return alpha_new, b, count_sv


"""Collection of Kernel Functions"""

def Gaussian_rbf(a, b, sigma):
    """Gaussian RBF kernel for vector"""
    return np.exp(-1 * sigma * np.linalg.norm(a - b)**2)

def Polynomial(a, b, degree, const = 1):
    """Polynomial Kernle (ab + const)^degree"""
    return (np.dot(a.T, b) + const)**degree

def Sigmoid(a, b, alpha = 1, const = 0):
    """Hyperbolic Tangent Sigmoid Kernel """
    return np.tanh(alpha * np.dot(a.T, b) + const)

def Tensor_RBF_Kernel(Tx, Ty, sigma_vec):
    """
    CP-Tensor Gaussian RBF Kernel Function
    Input: Tx: a list of CP component for tensor Tx (Kruskal Tensor)
           Ty: a list of CP component for tensor Ty (Kruskal Tensor)
           sigma_vec: len ==  num_mode, scale parameter
    Return: kernel value of tensor Tx and Ty
    """
    k_val = 0
    num_mode = len(Tx)
    num_rank = Tx[0].shape[-1]
    for i in range(num_rank):
        for j in range(num_rank):
            tmp = 1.0
            for k in range(num_mode):
                    tmp = tmp * Gaussian_rbf(Tx[k][:, i], Ty[k][:, j], sigma_vec[k])
            k_val += tmp
    return k_val

def Tensor_Poly_Kernel(Tx, Ty, degree_vec, const = [1]):
    """
    CP-Tensor Gaussian Poly Kernel Function
    Input: Tx: a list of CP component for tensor Tx (Kruskal Tensor)
           Ty: a list of CP component for tensor Ty (Kruskal Tensor)
           degree_vec: len ==  num_mode, degree parameter
    Return: kernel value of tensor Tx and Ty
    """
    k_val = 0
    num_mode = len(Tx)
    num_rank = Tx[0].shape[-1]
    if len(const) == 1:
        const = const * num_mode 
    for i in range(num_rank):
        for j in range(num_rank):
            tmp = 1.0
            for k in range(num_mode):
                    tmp = tmp * Polynomial(Tx[k][:, i], Ty[k][:, j], degree_vec[k], const=const[k])
            k_val += tmp
    return k_val

def Tensor_Sigmoid_Kernel(Tx, Ty, alpha_vec, const = [1]):
    """
    CP-Tensor Gaussian RBF Kernel Function
    Input: Tx: a list of CP component for tensor Tx (Kruskal Tensor)
           Ty: a list of CP component for tensor Ty (Kruskal Tensor)
           alpha_vec: len ==  num_mode, multiplier
    Return: kernel value of tensor Tx and Ty
    """
    k_val = 0
    num_mode = len(Tx)
    num_rank = Tx[0].shape[-1]
    if len(const) == 1:
        const = const * num_mode
    for i in range(num_rank):
        for j in range(num_rank):
            tmp = 1.0
            for k in range(num_mode):
                    tmp = tmp * Sigmoid(Tx[k][:, i], Ty[k][:, j], alpha_vec[k], const[k])
            k_val += tmp
    return k_val

def Tensor_Kernel_Matrix(Tx, par_vec, kernel_fun = 'Gaussian', **kwargs):
    """Return n by n Kernel matrix K
    Tx: list of training data, len==n; each component is a Kruskal tensor
    par_vec: parameter len == num_mode
    kernel_fun: type of kernel function, default is Gaussian RBF
    """
    n = len(Tx)
    num_mode = len(Tx[0])
    if kernel_fun != 'Gaussian':
        if "const" not in kwargs:
            const = [1] * num_mode
        else:
            const = kwargs['const']
    K = np.zeros((n , n))
    for i in range(n):
        for j in range(i, n):
            if kernel_fun == 'Polynomial':
                K[i, j] = Tensor_Poly_Kernel(Tx[i], Tx[j], par_vec, const)
            elif kernel_fun == 'Sigmoid':
                K[i, j] = Tensor_Sigmoid_Kernel(Tx[i], Tx[j], par_vec, const)
            else:
                K[i, j] = Tensor_RBF_Kernel(Tx[i], Tx[j], par_vec)
            K[j, i] = K[i, j]
    return K

# #########################################################################################################################################
# Class for CP-STM Classifier

class CP_STM():
    def __init__(self, X, y,  num_rank = 1, loss_type = "Hinge", solver = "QP", Decomposition = True):
        """Define the classifier and specify loss type and solver type 
        Default is Hinge loss (L1) STM solved with Quadratic Programming
        X: training tensor
        y: labels
        num_rank= cp rank

        """
        self.loss = loss_type
        self.solver = solver
        self.rank = num_rank
        if Decomposition:
            self.Tx = []    # List of traning data 
            for a in X:
                _, b = parafac(a, self.rank, return_errors= False)
                self.Tx.append(b)
        else:
            self.Tx = X
        self.Decomposition = Decomposition
        self.y = np.array(y)
        self.alpha = None
        self.b = None
        self.ST_count = None
        self.kernel_fun = None
        self.par_vec = None
        self.n = len(X)
        return
    
    def fit(self, kernel_fun, par_vec, C = None):
        """Fitting STM models"""
        self.kernel_fun = kernel_fun
        self.par_vec = par_vec
        t1 = time.time()
        K = Tensor_Kernel_Matrix(self.Tx, par_vec, kernel_fun=kernel_fun)
        # solve a QP model
        if self.solver == "QP":
            self.alpha, self.b, self.ST_count = SVM_QP(K, self.y, C) # Ususally C takes 1 / n_samples (equivalent to lambda = 1 / 2)
        else:
            self.alpha, self.b, self.ST_count = SVM_SQHinge(K, self.y, C)    #For Squared Hinge, C is lambda. Usually lambda = 1 / 2
        t2 = time.time()
        print("Training takes {} second".format(t2 - t1))
        return
    
    def predict(self, newX):
        Knew = np.zeros((self.n, len(newX)))
        for i in range(len(newX)):
            if self.Decomposition:
                _, tmpX = parafac(newX[i], self.rank, return_errors= False)
            else:
                tmpX = newX[i]
            for j in range(self.n):
                if self.kernel_fun == 'Polynomial':
                    Knew[j, i] = Tensor_Poly_Kernel(tmpX, self.Tx[j], self.par_vec)
                elif self.kernel_fun == 'Sigmoid':
                    Knew[j, i] = Tensor_Sigmoid_Kernel(tmpX, self.Tx[j], self.par_vec)
                else:
                    Knew[j, i] = Tensor_RBF_Kernel(tmpX, self.Tx[j], self.par_vec)
        Dy = np.diag(self.y)
        if self.solver == 'QP':
            f_val = np.dot(self.alpha, np.dot(Dy, Knew)) + self.b
        else:
            f_val = np.dot(self.alpha, np.dot(Dy, Knew)) + self.b   # STM model without offset parameter b (I choose to go with offset parameter.)
        sgn = lambda x: 1 if x >= 0 else -1
        y_pred = [sgn(f) for f in f_val]
        return y_pred
    
    def predict_performance(self, Xtest, ytest):
        ypred = self.predict(Xtest)
        return classification_metrics(ytest, ypred)


"""The Next part is for Random Projection based CP-STM"""

def Gaussian_Random_Matrix(P, D):
    """Generate a list of Gaussian Ranodm Matrix 
    len(P) == len(D)
    """
    P_MM = []
    for a, b in zip(P, D):
        P_MM.append(np.random.randn(a, b))
    return P_MM

def Apply_Projection(Tx, P_MM):
    """Apply rank-1 dense random projection to Kruskal tensors"""
    ans = []
    for a, b in zip(Tx, P_MM):
        ans.append(np.dot(b, a))
    return ans

"""A Superclass called RPSTM built on top of STM"""
class RPSTM(CP_STM):

    def __init__(self, X, y,  num_rank = 1, P = None,  Random_Projection = True, loss_type = "Hinge", solver = "QP", Decomposition = True, D = None):
        super().__init__(X, y, num_rank, loss_type, solver, Decomposition)
        if not Decomposition:
            self.D = D
        else:
            self.D = X[0].shape
        if Random_Projection == True:
            if not P:
                self.P = [int(0.7 * i) for i in self.D]
            else:
                self.P = P
            self.RPMM = Gaussian_Random_Matrix(self.P, self.D)
            self.Tx = [Apply_Projection(a, self.RPMM) / np.sqrt(np.prod(self.P)) for a in self.Tx]
        else:
            self.RPMM = None
            self.P = None
        return
    
    def fit(self, kernel_fun, par_vec, C = None):
        # Estimation part can use the base class model
        return super().fit(kernel_fun, par_vec, C)
    
    def predict(self, newX):
        Knew = np.zeros((self.n, len(newX)))
        for i in range(len(newX)):
            if self.Decomposition:
                _, tmpX = parafac(newX[i], self.rank, return_errors= False)
            else:
                tmpX = newX[i]
            if self.RPMM:
                tmpX = Apply_Projection(tmpX, self.RPMM)  / np.sqrt(np.prod(self.P))
            for j in range(self.n):
                if self.kernel_fun == 'Polynomial':
                    Knew[j, i] = Tensor_Poly_Kernel(tmpX, self.Tx[j], self.par_vec)
                elif self.kernel_fun == 'Sigmoid':
                    Knew[j, i] = Tensor_Sigmoid_Kernel(tmpX, self.Tx[j], self.par_vec)
                else:
                    Knew[j, i] = Tensor_RBF_Kernel(tmpX, self.Tx[j], self.par_vec)
        Dy = np.diag(self.y)
        if self.solver == 'QP':
            f_val = np.dot(self.alpha, np.dot(Dy, Knew)) + self.b
        else:
            f_val = np.dot(self.alpha, np.dot(Dy, Knew)) + self.b
        sgn = lambda x: 1 if x >= 0 else -1
        y_pred = [sgn(f) for f in f_val]
        return y_pred
    
    def predict_performance(self, Xtest, ytest):
        ypred = self.predict(Xtest)
        return classification_metrics(ytest, ypred)


"""A Superclass called Tensor Ensemble Classifier Built on top of RPSTM"""
class TEC():
    # Tensor Ensemble classifier for big data

    def __init__(self, X, y, P = None, num_rank = 1, Random_Projection = True, num_ensemble = 1, loss_type = "Hinge", solver = "QP", Decomposition = True, D = None):
        self.ensemble = num_ensemble
        self.classifier = []
        for i in range(self.ensemble):
            self.classifier.append(RPSTM(X, y, P = P, num_rank = num_rank, Random_Projection = Random_Projection, loss_type = loss_type, solver = solver, Decomposition = Decomposition, D=D))
        return
    
    def fit(self, kernel_fun, par_vec, C = None):
        """Fit the Ensemble model without returning any information. There might be more models and returning all would be messy"""
        for model in self.classifier:
            model.fit(kernel_fun, par_vec, C)
        return 

    def predict(self, Xtest):
        ans = np.zeros((len(Xtest), self.ensemble))
        for j in range(ans.shape[1]):
            ans[:, j] = self.classifier[j].predict(Xtest)
        ans = np.mean(ans, axis=1)   # Each column is a prediction for all testing data in a single RPSTM. The overall is the average of all rows
        sgn = lambda x: 1 if x >= 0 else -1
        ypred = [sgn(f) for f in ans]
        return ypred
    

    

"""updated version has three classes including all the components"""


        





  

     
