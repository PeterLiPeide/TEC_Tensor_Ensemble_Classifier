#%%
import numpy as np 
import tensorly as tl 
import CP_STM as STM 
from classification_tools import classification_metrics
from sklearn.model_selection import train_test_split
from hdf5storage import loadmat
import pandas as pd 
from tensorly.decomposition import parafac


Df = loadmat('C:\\Users\\kira2\\Desktop\\TEC_Full_Code\\Simu_Df_F1.mat')
X1 = Df['Class_1']
X2 = Df['Class_2']
X1 = [np.squeeze(X1[i, :, :, :]) for i in range(100)]
X2 = [np.squeeze(X2[i, :, :, :]) for i in range(100)]
X = X1 + X2

Xnew = []
for a in X:
    _, b = parafac(a, 1, return_errors= False)
    Xnew.append(b)


y = [-1 for i in range(100)] + [1 for i in range(100)]    # label two classes as -1 and 1
print(len(Xnew))
print(len(y))


#%%
"""Perform Classification"""
accuracy_Hinge = []
accuracy_SQHinge = []
accuracy_TEC = []
accuracy_TEC_SQ = []
for i in range(100):
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xnew, y, test_size = 0.2, stratify = y)
    m1 = STM.RPSTM(Xtrain, ytrain, num_rank=1, P = [45, 45, 45], Decomposition=False, D = [50, 50, 50])
    m2 = STM.RPSTM(Xtrain, ytrain, num_rank=1, P = [45, 45, 45], loss_type="SQHinge", solver='Gaussian', Decomposition=False, D = [50, 50, 50])
    m3 = STM.TEC(Xtrain, ytrain, P = [45, 45, 45], num_rank=1, Random_Projection=True, num_ensemble=3, Decomposition=False, D = [50, 50, 50])
    m4 = STM.TEC(Xtrain, ytrain, P = [45, 45, 45], num_rank=1, Random_Projection=True, loss_type='SQHinge', solver='Gaussian', num_ensemble=3, Decomposition=False, D = [50, 50, 50])
    m1.fit('Gaussian', [0.01, 0.5, 0.1], 1 / 160)
    m2.fit('Gaussian', [0.01, 0.5, 0.1], 0.5)  # Notice the C parameter has different meaning in SQHinge and Hinge Solver
    m3.fit('Gaussian', [0.01, 0.5, 0.1], 1 / 160)
    m4.fit('Gaussian', [0.01, 0.5, 0.1], 0.5)
    pred1 = m1.predict(Xtest)
    pred2 = m2.predict(Xtest)
    pred3 = m3.predict(Xtest)
    pred4 = m4.predict(Xtest)
    a1, _, _, _ = classification_metrics(ytest, pred1)
    a2, _, _, _ = classification_metrics(ytest, pred2)
    a3, _, _, _ = classification_metrics(ytest, pred3)
    a4, _, _, _ = classification_metrics(ytest, pred4)
    accuracy_Hinge.append(a1)
    accuracy_SQHinge.append(a2)
    accuracy_TEC.append(a3)
    accuracy_TEC_SQ.append(a4)

#%%
df = pd.DataFrame({'RPSTM1' : accuracy_Hinge, 'RPSTM2': accuracy_SQHinge, 'TEC1' : accuracy_TEC, 'TEC2' : accuracy_TEC_SQ})
df.to_csv('F1_Result.csv', index=False)
# %%
