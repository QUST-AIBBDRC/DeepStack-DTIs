import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.preprocessing import scale,StandardScaler

data_=pd.read_csv(r'GPCR.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)

lgb_model=LGBMRegressor()
lgbresult1=lgb_model.fit(shu,label.ravel())
feature_importance=lgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:200]
train_data=shu[:,mask]

X = train_data
data_csv=pd.DataFrame(data=X)
data_csv.to_csv('GPCR_LGB.csv')
