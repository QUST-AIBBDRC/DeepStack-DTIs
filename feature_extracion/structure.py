import numpy as np
import pandas as pd

ALPHABET='ACDEFGHIKLMNPQRSTVWY'
file='s0.spd33'

def read_spd_file(file):
    with open(file) as f:
         records=f.read()
    record=records.split('\n')[1:]
    return record

def SSC(file):
    spd_data=read_spd_file(file)
    C_count,E_count,H_count= 0,0,0
    header=[]
    for f in range(3):
        header.append('SSC.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    L_sequence=0
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        matrix = seq.split()
        if matrix[2] == 'C':
           C_count += 1
        if matrix[2] == 'H':
           H_count += 1
        if matrix[2] == 'E':
           E_count += 1   
    C_count = C_count/L_sequence
    E_count = E_count/L_sequence
    H_count = H_count/L_sequence 
    vector=[C_count,H_count]
    for v in vector:
        vec.append(v)
    result.append(vec)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=SSC(file)
    vector.append(vector1)    
shu=np.array(vector)
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('ssc_spd_GPCR.csv',header=False,index=False)   

def ASA(file):
    spd_data=read_spd_file(file)
    header=[]
    for f in range(1):
        header.append('ASA.'+str(f))
    result=[]
    result.append(header)
    ASA_sum =0
    L_sequence=0
    for z in range(len(spd_data)):
        seq=spd_data[z]
        L_sequence+=1
        if len(seq)==0:
           break
        matrix = seq.split()
        ASA_sum += float(matrix[3])
    vector = [ASA_sum/L_sequence]
    result.append(vector)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=ASA(file)
    vector.append(vector1)    
shu=np.array(vector)
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('asa_spd_GPCR.csv',header=False,index=False)   

def TAC(file):
    spd_data=read_spd_file(file)
    header=[]
    for f in range(8):
        header.append('TAC.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    psi_sin_sum,psi_cos_sum,phi_sin_sum,phi_cos_sum=0,0,0,0
    theta_sin_sum,theta_cos_sum=0,0
    tau_sin_sum,tau_cos_sum=0,0
    L_sequence=0
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        matrix = seq.split()
        phi_sin_sum += np.sin(float(matrix[4]))
        phi_cos_sum += np.cos(float(matrix[4]))
        psi_sin_sum += np.sin(float(matrix[5]))
        psi_cos_sum += np.sin(float(matrix[5]))
        theta_sin_sum += np.sin(float(matrix[6]))
        theta_cos_sum += np.sin(float(matrix[6]))
        tau_sin_sum += np.sin(float(matrix[7]))
        tau_cos_sum += np.cos(float(matrix[7]))
    phi_sin = phi_sin_sum/L_sequence
    phi_cos = phi_cos_sum / L_sequence
    psi_sin = psi_sin_sum / L_sequence
    psi_cos = psi_cos_sum / L_sequence
    theta_sin = theta_sin_sum/ L_sequence
    theta_cos = theta_cos_sum /L_sequence
    tau_sin = tau_sin_sum/L_sequence
    tau_cos = tau_cos_sum / L_sequence  
    vector = [phi_sin, phi_cos , psi_sin ,psi_cos ,theta_sin, theta_cos,tau_sin,tau_cos]
    for v in vector:
        vec.append(v)
    result.append(vec)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=TAC(file)
    vector.append(vector1)    
shu=np.array(vector)
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('tac_spd_GPCR.csv',header=False,index=False)   

def TAAC(file): 
    spd_data=read_spd_file(file)
    header=[]
    for f in range(80):
        header.append('TAAC.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    L_sequence=0
    i = 0
    feature = []
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        seq=seq.split()
        spd_a=seq[3:]   
        feature.append(spd_a)
    matrix = [[0 for x in range(8)] for y in range(10)]
    degree_matrix = [[0 for x in range(8)] for y in range(L_sequence-1)]
    degree_index = 0
    for x in range(0,L_sequence-1):
        degree_index = 0
        for y in range(1,5):
            degree_matrix[x][degree_index] = np.math.sin(float(feature[x][y]) * np.pi / 180 )
            degree_matrix[x][degree_index+1] = np.math.cos(float(feature[x][y]) * np.pi / 180 )
            degree_index += 2
    array = degree_matrix
    for k in range(0, 10):
        for j in range(0, 8):
            for i in range(0, L_sequence - 1 - k):
                matrix[k][j] += float(array[i][j]) * float(array[i + k][j])
                matrix[k][j] = matrix[k][j] / (L_sequence - 1)
    matrix=np.array(matrix)
    vector=np.reshape(matrix,(1,matrix.shape[0]*matrix.shape[1]))
    vector1=np.array(vector).astype(float)
    vector2=vector1.T
    vector3=vector2[:,0]
    for v in vector3:
        vec.append(v)
    result.append(vec)
    return vector,result
top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=TAAC(file)
    vector.append(vector1)    
shu=np.array(vector)
shu=np.reshape(shu,[shu.shape[0],shu.shape[2]])
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('taac_spd_GPCR.csv',header=False,index=False)   

def SPAC(file): 
    spd_data=read_spd_file(file)
    header=[]
    for f in range(30):
        header.append('SPAC.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    L_sequence=0
    feature = []
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        seq=seq.split()
        spd_a=seq[3:]   
        feature.append(spd_a)
    probability_matrix = [[0 for x in range(3)] for y in range(10)]
    temp_feature = feature
    for k in range(0, 10):
        for j in range(5, 8):
            for i in range(0, L_sequence - 1 - k):
                probability_matrix[k][j - 5] += np.float(temp_feature[i][j]) * np.float(temp_feature[i + k][j])
                probability_matrix[k][j - 5] = probability_matrix[k][j - 5] / (L_sequence - 1)
    probability_matrix=np.array(probability_matrix)
    vector=np.reshape(probability_matrix,(1,probability_matrix.shape[0]*probability_matrix.shape[1]))
    vector1=np.array(vector).astype(float)
    vector2=vector1.T
    vector3=vector2[:,0]
    for v in vector3:
        vec.append(v)
    result.append(vec)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=SPAC(file)
    vector.append(vector1)    
shu=np.array(vector)
shu=np.reshape(shu,[shu.shape[0],shu.shape[2]])
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('spac_spd_GPCR.csv',header=False,index=False)   

def TAB(file):
    spd_data=read_spd_file(file)
    header=[]
    for f in range(64):
        header.append('TAB.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    L_sequence=0
    i = 0
    feature = []
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        seq=seq.split()
        spd_a=seq[3:]   
        feature.append(spd_a)      
    array = feature
    degree_matrix = [[0 for x in range(8)] for y in range(L_sequence-1)]
    matrix = [[0 for x in range(8)] for y in range(8)]
    degree_index = 0
    for x in range(0,L_sequence-1):
        degree_index = 0
        for y in range(1,5):
            degree_matrix[x][degree_index] = np.math.sin(float(array[x][y]) * np.pi / 180 )
            degree_matrix[x][degree_index+1] = np.math.cos(float(array[x][y]) * np.pi / 180  )
            degree_index += 2
    array = degree_matrix
    L_sequence -= 1
    for k in range(0, 8):
        for l in range(0, 8):
            for i in range(0,L_sequence - 1 ):
                matrix[k][l] += float(array[i][k]) * float(array[i + 1][l])
                matrix[k][l] = matrix[k][l] / (L_sequence - 1)
    matrix=np.array(matrix)
    vector=np.reshape(matrix,(1,matrix.shape[0]*matrix.shape[1]))
    vector1=np.array(vector).astype(float)
    vector2=vector1.T
    vector3=vector2[:,0]
    for v in vector3:
        vec.append(v)
    result.append(vec)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=TAB(file)
    vector.append(vector1)    
shu=np.array(vector)
shu=np.reshape(shu,[shu.shape[0],shu.shape[2]])
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('tab_spd_GPCR.csv',header=False,index=False)   

#vector,result=[],[]
#vector,result=TAB(file)

def SPB(file):
    spd_data=read_spd_file(file)
    header=[]
    for f in range(9):
        header.append('SPB.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    L_sequence=0
    i = 0
    feature = []
    for z in range(len(spd_data)):
        seq=spd_data[z]
        if len(seq)==0:
           break
        L_sequence+=1
        seq=seq.split()
        spd_a=seq[3:]   
        feature.append(spd_a)
    probability_matrix = [[0 for x in range(3)] for y in range(3)]      
    array = feature
    for k in range(0, 3):
        for l in range(5, 8):
            for i in range(0, L_sequence- 1 ):
                probability_matrix[k][l- 5] += np.float(array[i][k]) * np.float(array[i + 1][l -5 ])
                probability_matrix[k][l - 5] = probability_matrix[k][l - 5] / (L_sequence - 1)
    probability_matrix=np.array(probability_matrix)
    vector=np.reshape(probability_matrix,(1,probability_matrix.shape[0]*probability_matrix.shape[1]))
    vector1=np.array(vector).astype(float)
    vector2=vector1.T
    vector3=vector2[:,0]
    for v in vector3:
        vec.append(v)
    result.append(vec)
    return vector,result

top='s'
tail='.spd33'
vector,result=[],[]
for i in range(95):
    file=top+str(i)+tail
    vector1,result1=SPB(file)
    vector.append(vector1)    
shu=np.array(vector)
shu=np.reshape(shu,[shu.shape[0],shu.shape[2]])
csv_data=pd.DataFrame(data=shu)
csv_data.to_csv('spb_spd_GPCR.csv',header=False,index=False)   