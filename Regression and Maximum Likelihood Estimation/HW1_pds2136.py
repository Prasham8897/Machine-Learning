#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = "hw1_data\\"
X_train = pd.read_csv(path + "X_train.csv", header=None).values
X_test = pd.read_csv(path + "X_test.csv", header=None).values
y_train = pd.read_csv(path + "y_train.csv", header=None).values
y_test = pd.read_csv(path + "y_test.csv", header=None).values


# In[3]:


Wrr = []
df_lambdas = []
lambda_values = range(5001)


# In[4]:


for lambda_value in lambda_values:
    XTX = np.matmul(X_train.transpose(),X_train)
    lambdaI = lambda_value * np.identity(XTX.shape[0]) 
    XTy = np.matmul(X_train.transpose(),y_train)
    XTX_lambdaI_inv = np.linalg.inv(lambdaI + XTX)
    W = np.matmul(XTX_lambdaI_inv, XTy)
    df_lambda = np.trace(np.matmul(np.matmul(X_train,XTX_lambdaI_inv),X_train.transpose()))
    df_lambdas.append(df_lambda)
    Wrr.append([i[0] for i in W.tolist()])


# In[5]:


w = pd.DataFrame(Wrr)
w.columns= ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year made", "bias"]
w["df_lambda"] = df_lambdas


# In[6]:


plt.figure(figsize=(7,7))
for i in w.columns[:-1]:
    plt.plot(list(w["df_lambda"]),list(w[i]), label = i, marker = "o", markersize = 4,  linewidth=2)
plt.legend()
plt.xlabel("df(lambda)")
plt.show()


# # Prediction

# In[7]:


y=[]
Wrr=[]
lambda_values = range(51)
rmse_list = []
for lambda_value in lambda_values:
    XTX = np.matmul(X_train.transpose(),X_train)
    lambdaI = lambda_value * np.identity(XTX.shape[0]) 
    XTy = np.matmul(X_train.transpose(),y_train)
    XTX_lambdaI_inv = np.linalg.inv(lambdaI + XTX)
    W = np.matmul(XTX_lambdaI_inv, XTy)
    Wrr.append([i[0] for i in W.tolist()])
    y_predicted = np.matmul(X_test,W)
    y.append([i[0] for i in y_predicted.tolist()])
    rmse = np.sqrt(np.mean((y_predicted - y_test)**2))
    rmse_list.append(rmse)


# In[8]:


plt.figure(figsize=(5,5))
plt.plot(lambda_values,rmse_list)
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title("Effect of Lambda on RMSE(Test Data)")
plt.show()


# # pth-order polynomial regression

# In[12]:


p_values = [1,2,3]


# In[13]:


lambda_values = range(101)
rmses = pd.DataFrame()
for p in p_values:
    rmse_list = []
    X_temp_train = X_train
    X_temp_test = X_test
    if(p!=1):
        for j in range(2,p+1):
            for k in range(6):
                mean = np.mean(np.array(X_temp_train[:,k])**j)
                std = np.sqrt(np.var(np.array(X_temp_train[:,k])**j))
                X_temp_train = np.append(X_temp_train, np.array(np.array(np.array(X_temp_train[:,k]**j) - mean)/std)
                                         .reshape(np.array((np.array(X_temp_train[:,k]**2) - mean)).shape[0], 1),1)
                X_temp_test = np.append(X_temp_test, np.array(np.array(np.array(X_temp_test[:,k]**j) - mean)/std)
                                        .reshape(np.array((np.array(X_temp_test[:,k]**2) - mean)).shape[0], 1),1)
                
#     print(X_temp_test.shape)
#     pd.DataFrame(X_temp_train).to_csv("training data" + str(p))
    for lambda_value in lambda_values:
        XTX = np.matmul(X_temp_train.transpose(),X_temp_train)
        lambdaI = lambda_value * np.identity(XTX.shape[0]) 
        XTy = np.matmul(X_temp_train.transpose(),y_train)
        XTX_lambdaI_inv = np.linalg.inv(lambdaI + XTX)
        W = np.matmul(XTX_lambdaI_inv, XTy)
        
        y_predicted = np.matmul(X_temp_test,W)
        rmse = np.sqrt(np.mean((y_predicted - y_test)**2))
        rmse_list.append(rmse)
    rmses[p] = rmse_list    


# In[11]:


plt.figure(figsize=(5,5))
for i in rmses.columns:
    plt.plot(lambda_values,rmses[i], label = ("p = " + str(i)))
plt.legend()
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title("Effect of Lambda and order of Polynomial(p) on RMSE(Test Data)")
plt.show()


# In[ ]:





