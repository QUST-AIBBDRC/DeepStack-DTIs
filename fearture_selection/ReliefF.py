import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from skrebate import ReliefF

def ReliefF_Method(X,y,n):
    X=np.array(X)
    y=np.array(y)
#    y = y[:, 0]
    clf = ReliefF(n_features_to_select=n, n_neighbors=50)
    Reresult = clf.fit_transform(X,y)
    Reresult = pd.DataFrame(Reresult)
    Reresult.to_csv("GPCR_ReliefF.csv")
    return None

data_=pd.read_csv(r'GPCR.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
result=ReliefF_Method(shu,label,200)
