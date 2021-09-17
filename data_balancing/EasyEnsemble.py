import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler 
from imblearn.ensemble import EasyEnsemble

data_=pd.read_csv(r'GPCR.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((635,1))
label2=np.zeros((20550,1))
label=np.append(label1,label2)

X=data
y=label

ee = EasyEnsemble(random_state=0, n_subsets=10)
X_resampled, y_resampled = ee.fit_sample(X, y)

shu2 =X_resampled
shu3 =y_resampled
data_csv = pd.DataFrame(data=shu2)
data_csv.to_csv('GPCR_EE.csv')
data_csv = pd.DataFrame(data=shu3)
data_csv.to_csv('GPCR_EE_label.csv')