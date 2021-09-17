import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler
import utils.tools as utils
from L1_Matine import mutual_mutual

data_=pd.read_csv(r'GPCR.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
data_2=mutual_mutual(shu,label,k=200)
shu1=data_2
data_csv = pd.DataFrame(data=shu1)
data_csv.to_csv('GPCR_MI.csv')