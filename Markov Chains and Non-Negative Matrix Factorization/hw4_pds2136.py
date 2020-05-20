#!/usr/bin/env python
# coding: utf-8

# # Problem 1 (Markov chains)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("CFB2019_scores.csv",header=None)


# In[3]:


data["winA"] = [1 if data[1][i] > data[3][i] else 0 for i in range(len(data))]
data["winB"] = [1 if data[1][i] < data[3][i] else 0 for i in range(len(data))]


# In[4]:


M_hat = np.zeros(shape=(769,769))


# In[5]:


for _,row in data.iterrows():
    i = row[0] - 1
    j = row[2] - 1
    pointsA = row[1]
    pointsB = row[3]
    winA = row["winA"]
    winB = row["winB"]
    incA = (pointsA / (pointsA + pointsB))
    incB = (pointsB / (pointsA + pointsB))
    M_hat[i][i]+=(winA + incA)
    M_hat[j][j]+=(winB + incB)
    M_hat[i][j]+=(winB + incB)
    M_hat[j][i]+=(winA + incA)


# In[6]:


row_sums = np.sum(M_hat,axis=1)
M = M_hat / row_sums[:, np.newaxis]
M1 = M


# In[7]:


# w_0 = np.full(shape=(1,769),fill_value = 1/769)
np.random.seed(0)
w_0 = np.random.uniform(0,1,size=769)
w_0 /= np.sum(w_0)
# w_0 = w_0.reshape(1,769)


# In[2]:


w, v = np.linalg.eig(M.T)
index = np.argmax(w)
v_1 = v[:,index]
# .reshape(769,1)
# print("The value of Eigen value is " + str(round(w[index],2)))
# print("The value of V_transpose * V is " + str(np.matmul(v_1.T,v_1)[0][0]))


# In[9]:


w_inf = np.real(v_1.T/np.sum(v_1))


# In[11]:


plot1 = []
T = [10,100,1000,10000]
values = dict()
positions = dict()
for i in range(1,10001):
    temp = np.matmul(w_0,M)
    w_0 = temp
    plot1.append(np.linalg.norm((temp - w_inf),ord=1))
    if i in T:
        values[i] = temp


# In[12]:


plt.plot(range(1,10001),plot1)
plt.title("Plot of the first order norm of ||w_t - w_inf|| as a function of t")
plt.xlabel("t")
plt.ylabel("value of the norm")
plt.show()


# In[13]:


for i in list(values.keys()):
    positions[i] = np.argsort(-values[i])
    positions[i] = positions[i].tolist()[:25]
    values[i] = -np.sort(-values[i])
    values[i] = values[i].tolist()[:25]


# In[14]:


filename = "TeamNames.txt"
names=[]
for i in open(filename,"r"):
    names.append(i.strip())
names=np.array(names)


# In[15]:


for t in T:
    print("-------------------------------------------------------------------------")
    print("Top 25 teams for t = " + str(t))
    print("-------------------------------------------------------------------------")
    temp = pd.DataFrame({"Name":names[positions[t]].tolist(),"Value in W_t":values[t]})
    print(temp)
    print("-------------------------------------------------------------------------")


# # Problem 2 (Nonnegative matrix factorization)

# In[96]:


N = 3012
M = 8447


# In[97]:


num_iter = 100
rank = 25


# In[98]:


X = np.zeros(shape=(N,M))
np.random.seed(0)
W = np.random.uniform(1,2,size=(N,rank))
H = np.random.uniform(1,2,size=(rank,M))


# In[99]:


filename = "nyt_data.txt"


# In[100]:


doc_index = 0
for i in open(filename,"r"):
    row = (i.strip().split(","))
    for element in row:
        temp = element.split(":")
        X[int(temp[0])-1][doc_index]= int(temp[1])
    doc_index+=1


# In[101]:


objective = []
for run in range(num_iter):
    X_WH = np.divide(X,(np.matmul(W,H)+10**-16))
    W_T = W.T
    row_sums = np.sum(W_T,axis=1)
    normalized_W_T =  W_T / (row_sums[:, np.newaxis] + 10**-16)
    second_term = np.matmul(normalized_W_T,X_WH)
    H = np.multiply(H,(second_term))
    
    X_WH = np.divide(X,(np.matmul(W,H)+10**-16))
    H_T = H.T
    col_sums = np.sum(H_T,axis=0)
    normalized_H_T =  H_T / (col_sums + 10**-16)
    second_term = np.matmul(X_WH,normalized_H_T)
    W = np.multiply(W,(second_term))
    
    WH = np.matmul(W,H)
    log_WH = np.log(WH+ 10**-16) 
    temp = np.multiply(X,log_WH) - WH
    objective_value = -np.sum(temp)
    objective.append(objective_value)


# In[103]:


plt.plot(range(1,101),objective)
plt.title("Plot for objective")
plt.xlabel("Iterations")
plt.ylabel("value of the objective")
plt.show()


# In[104]:


col_sums = np.sum(W,axis=0)
normalized_W =  W / (col_sums + 10**-16)


# In[108]:


filename = "nyt_vocab.dat"
names=[]
for i in open(filename,"r"):
    names.append(i.strip())
names=np.array(names)


# In[116]:


positions = dict()
values = dict()
list_topic_word = dict()
for i in range(rank):
    positions[i] = np.argsort(-W[:,i])
    positions[i] = positions[i].tolist()[:10]
    positions[i] = [names[j] for j in positions[i]]
    values[i] = -np.sort(-W[:,i])
    values[i] = values[i].tolist()[:10]


# In[159]:


answer = dict()
for i in range(25):
    text=""
    for j in range(10):
        text+= positions[i][j] + " : " +str(values[i][j]) + "\n"
    answer[i]=text[:-1]


# In[160]:


for i in range(25):
    print(i+1)
    print("---------------------------")
    print(answer[i])
    print("---------------------------")
    


# In[ ]:




