"""
Synthetic Tensor Generator for paper 
Li, P., Karim, R., & Maiti, T. (2021). TEC: Tensor Ensemble Classifier for Big Data. arXiv preprint arXiv:2103.00025.

This module is to generate synthetic data in our simulation study.
"""
#%%
import numpy as np 

from tensorly.tenalg import multi_mode_dot
from hdf5storage import savemat


'''Basic tool function to generate ranodm matrices'''

def rvs(dim=3):
    """Generate random Orthogonal matrix"""
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

def AR_Matrix(rho, n):
    """Generate Auto-Regressive matrix"""
    H = np.zeros((n ,n))
    for i in range(n):
        for j in range(n):
            H[i, j] = rho ** abs(i - j)
    return H

def Martingale_Matrix(n):
    """Generate Brownian Motion type of covariance matrix"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = min(i + 1, j + 1) / n
    return H


'''Generate Tensors According to the Description in Our Paper'''
def F1_Model(mean):
    '''Generate data from F1 Model'''
    mode1 = np.random.normal(mean, 1, 50)
    mode2 = np.random.normal(mean, 1, 50)
    mode3 = np.random.normal(mean, 1, 50)
    t1 = np.tensordot(mode1, mode2, axes= 0)
    t2 = np.tensordot(t1, mode3, axes= 0)
    return t2

def F2_Model(mean, Num_rank, cov1, cov2, cov3, cov4):
    '''Generate data from F2 Model'''
    total = 0
    for i in range(Num_rank):
        mode1 = np.random.multivariate_normal(mean, cov1)
        mdoe2 = np.random.multivariate_normal(mean, cov2)
        mode3 = np.random.multivariate_normal(mean, cov3)
        mode4 = np.random.multivariate_normal(mean,cov4)
        temp = np.tensordot(mode1, mdoe2, axes=0)
        temp = np.tensordot(temp, mode3, axes=0)
        temp = np.tensordot(temp, mode4, axes=0)
        total += temp
    return total

def F3_Model(mean, Num_rank, cov1, cov2, cov3, cov4):
    '''Generate data from F3 Model'''
    total = 0
    for i in range(Num_rank):
        mode1 = np.random.multivariate_normal(mean, cov1)
        mdoe2 = np.random.multivariate_normal(mean, cov2)
        mode3 = np.random.multivariate_normal(mean, cov3)
        mode4 = np.random.multivariate_normal(mean,cov4)
        temp = np.tensordot(mode1, mdoe2, axes=0)
        temp = np.tensordot(temp, mode3, axes=0)
        temp = np.tensordot(temp, mode4, axes=0)
        total += temp
    return total

def F4_Model(mean):
    mode1 = np.random.gamma(mean, 2, 50)
    mode2 = np.random.multivariate_normal(np.ones(50), np.eye(50))
    mode3 = np.random.uniform(mean , mean + 1, 50)
    tmp = np.tensordot(mode1, mode2, axes=0)
    tmp = np.tensordot(tmp, mode3, axes=0)
    return tmp


def F5_Model(mean):
    mode1 = np.random.gamma(mean, 2, 50)
    mode2 = np.random.multivariate_normal(np.ones(50), np.eye(50))
    mode3 = np.random.gamma(2, 1, 50)
    mode4 = np.random.uniform(mean - 0.5, mean + 0.5, 50)
    tmp = np.tensordot(mode1, mode2, axes=0)
    tmp = np.tensordot(tmp, mode3, axes=0)
    tmp = np.tensordot(tmp, mode4, axes=0)
    return tmp

def M1_Model(mean):
    return np.random.randn(50, 50, 50) + mean

def T1_Model(mean, cov1, cov2, cov3):
    g_core = np.random.randn(50, 50, 50) + mean
    return multi_mode_dot(g_core, [cov1, cov2, cov3])

def T2_Model(mean, cov1, cov2, cov3, cov4):
    g_core = np.random.randn(50, 50, 50, 50) + mean
    return multi_mode_dot(g_core, [cov1, cov2, cov3, cov4])




"""Generating data and save as mat file for further use"""
# C1 = [F1_Model(0) for i in range(100)]
# C2 = [F1_Model(0.5) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_F1.mat", df)
# # %%

# # F2 Model
# cov1 = np.eye(50)
# cov2 = AR_Matrix(0.7, 50)
# cov3 = AR_Matrix(0.3, 50)
# cov4 = Martingale_Matrix(50)
# C1 = [F2_Model(np.zeros(50), 1, cov1, cov2, cov3, cov4) for i in range(100)]
# C2 = [F2_Model(np.ones(50), 1, cov1, cov2, cov3, cov4) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_F2.mat", df)

# # F3 model 
# C1 = [F3_Model(np.zeros(50), 3, cov1, cov2, cov3, cov4) for i in range(100)]
# C2 = [F3_Model(np.ones(50), 3, cov1, cov2, cov3, cov4) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_F3.mat", df)

# F4 model
C1 = [F4_Model(4) for i in range(100)]
C2 = [F4_Model(6) for i in range(100)]
df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
savemat("Simu_Df_F4.mat", df)

# F5 Model
C1 = [F5_Model(4) for i in range(100)]
C2 = [F5_Model(5) for i in range(100)]
df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
savemat("Simu_Df_F5.mat", df)

# # M1 Model
# C1 = [M1_Model(0) for i in range(100)]
# C2 = [M1_Model(0.5) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_M1.mat", df)

# # T1 Model 
# cov5 = rvs(50)
# C1 = [T1_Model(0, cov1, cov5, cov2) for i in range(100)]
# C2 = [T1_Model(0.5, cov1, cov5, cov2) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_T1.mat", df)


# # T2 Model
# C1 = [T2_Model(0, cov1, cov5, cov2, cov4) for i in range(100)]
# C2 = [T2_Model(0.5, cov1, cov5, cov2, cov4) for i in range(100)]
# df = {"Class_1" : np.array(C1), "Class_2" : np.array(C2)}
# savemat("Simu_Df_T2.mat", df)
