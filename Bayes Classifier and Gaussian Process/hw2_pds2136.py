#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


# # Problem-2 Classification
# 
# #### 1- Naive Bayes,  2- KNN

# In[2]:


X = pd.read_csv("Bayes_classifier\\X.csv",header=None).values
y = pd.read_csv("Bayes_classifier\\y.csv",header=None).values
kf = KFold(n_splits=10,shuffle=True,random_state=0)


# In[3]:


def compute_confusion_matrix(y_test,y_prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0    
    for i in range(len(y_test)):
        if(y_prediction[i] == y_test[i]):
            if(y_prediction[i] == 0):
                TN+=1
            else:
                TP+=1
        else:
            if(y_prediction[i] == 0):
                FN+=1
            else:
                FP+=1
    return [TP,FP,FN,TN]


# ####  Naive Bayes

# In[4]:


TP = []
FP = []
TN = []
FN = []
lambda0=[]
lambda1=[]
for train_index,test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    
    py0 = len(np.where(y_train == 0)[0])/len(X_train)
    py1 = len(np.where(y_train == 1)[0])/len(X_train)
    
    lambda_0_values = []
    lambda_1_values = []
    
    for j in range(len(X_train[0])):
        lambda0_numerator = 1
        lambda0_denominator = 1
        lambda1_numerator = 1
        lambda1_denominator = 1

        for i in range(len(X_train)):
            lambda1_numerator+=(y_train[i] * X_train[i][j])
            lambda1_denominator+=(y_train[i])
     
            lambda0_numerator+=((1-y_train[i]) * X_train[i][j])
            lambda0_denominator+=(1-y_train[i])
        lambda_0_values.append((lambda0_numerator/lambda0_denominator)[0])
        lambda_1_values.append((lambda1_numerator/lambda1_denominator)[0])
        
    lambda0.append(lambda_0_values)
    lambda1.append(lambda_1_values)
    y_predictions = []
    for i in range(len(X_test)):
        
        y_prob = []
        product_0 = 1
        product_1 = 1

        for j in range(len(X_test[0])):
            poisson_prob_0 = poisson.pmf(X_test[i][j],lambda_0_values[j])
            poisson_prob_1 = poisson.pmf(X_test[i][j],lambda_1_values[j])
            
            product_0*=poisson_prob_0
            product_1*=poisson_prob_1
        
        y_prob.append(product_0*py0)
        y_prob.append(product_1*py1)

        y_predictions.append(np.argmax(y_prob))
        
    results = compute_confusion_matrix(y_test,y_predictions)
    
    TP.append(results[0])
    FP.append(results[1])
    FN.append(results[2])
    TN.append(results[3])
    


# In[5]:


confusion_matrix = [[sum(TP),sum(FP)],[sum(FN),sum(TN)]]
print("The Confusion Matrix is:")

print(confusion_matrix[0])
print(confusion_matrix[1])

print("The Accuracy is:", ((confusion_matrix[0][0] + confusion_matrix[1][1]) /
                           (confusion_matrix[0][0] + confusion_matrix[0][1] +confusion_matrix[1][0] + confusion_matrix[1][1])))


# In[6]:


avg_poisson_parameter_0 = np.mean(lambda0, axis=0)
avg_poisson_parameter_1 = np.mean(lambda1, axis=0)


# In[23]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
ax[0].stem(range(1,55),avg_poisson_parameter_0,"blue", markerfmt="bo")
ax[0].set_title("Stem plot of Poisson Parameters (Class 0: Not Spam)")
ax[0].set_xlabel("feature number")
ax[0].set_ylabel("value of lambda")
ax[0].set_xticks(np.arange(1,55, step=1))

ax[1].stem(range(1,55),avg_poisson_parameter_1,"red", markerfmt="ro")
ax[1].set_title("Stem plot of Poisson Parameters (Class 1: Spam)")
ax[1].set_xlabel("feature number")
ax[1].set_ylabel("value of lambda")
ax[1].set_xticks(np.arange(1,55, step=1))
plt.show()


# In[21]:


plt.figure(figsize=(15, 5))
plt.stem(range(1,55), avg_poisson_parameter_0, "blue", markerfmt="bo", label="0 Class: Not Spam")
plt.stem(range(1,55), avg_poisson_parameter_1, "red", markerfmt="ro", label="1 Class: Spam")
plt.title("Stem plot of Poisson Parameters (seperated by classes)")
plt.xlabel("Feature number")
plt.ylabel("Value of Lambda")
plt.xticks(np.arange(1,55, step=1))
plt.legend()
plt.show()


# #### KNN

# In[140]:


TP = []
FP = []
TN = []
FN = []
all_predictions=[]
all_labels = []
for train_index,test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    all_labels.append([i[0] for i in y_test])
    tp = np.zeros(shape = (20))
    fp = np.zeros(shape = (20))
    fn = np.zeros(shape = (20))
    tn = np.zeros(shape = (20))
    mean = np.mean(X_train,axis=0)
    std = np.std(X_train,axis=0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    predictions=[]
    for row in range(len(X_test)):
        distances=[]
        #computing the distance matrix
        for j in X_train:
            distances.append(np.sum(np.abs(X_test[row]-j)))
#       Finding 20 Nearest neighbours of the test sample. 
#       list_neighbours = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][:20]
        list_neighbours = np.argsort(distances)[:20]
        pred = []
        for k in range(1,21):
            count_1 = len(np.where(y_train[list_neighbours[:k]] == 1)[0])
            count_0 = len(np.where(y_train[list_neighbours[:k]] == 0)[0])
            # picking the nearest neighbour's label in the case of tie.
            if(count_1==count_0):
                temp_prediction = y_train[list_neighbours[0]][0]
                pred.append(temp_prediction)
                if(temp_prediction == y_test[row][0]):
                    if(temp_prediction == 1):
                        tp[k-1]+=1
                    else:
                        tn[k-1]+=1
                else:
                    if(temp_prediction == 1):
                        fp[k-1]+=1
                    else:
                        fn[k-1]+=1
            elif(count_1>count_0):
                temp_prediction = 1
                pred.append(temp_prediction)
                if(temp_prediction == y_test[row][0]):
                    tp[k-1]+=1
                else:
                    fp[k-1]+=1
            else:
                temp_prediction = 0
                pred.append(temp_prediction)
                if(temp_prediction == y_test[row][0]):
                    tn[k-1]+=1
                else:
                    fn[k-1]+=1
        predictions.append(pred)
    all_predictions.append(predictions)
    TP.append(list(tp))
    FP.append(list(fp))
    TN.append(list(tn))
    FN.append(list(fn))


# In[142]:


TP_list = (list(np.sum(TP,axis=0)))
FP_list = (list(np.sum(FP,axis=0)))
FN_list = (list(np.sum(FN,axis=0)))
TN_list = (list(np.sum(TN,axis=0)))
prediction_accuracies_cumulative = []
for i in range(20):
    prediction_accuracies_cumulative.append((TP_list[i] + TN_list[i])/
                                            (TP_list[i] + TN_list[i] + FP_list[i] + FN_list[i]))


# In[143]:


plt.plot(range(1,21),prediction_accuracies_cumulative)
plt.scatter(range(1,21),prediction_accuracies_cumulative)
plt.title("Accuracy v/s K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,21, step=1))
plt.show()


# # Problem-3 : Gaussian Process

# In[69]:


X_train = pd.read_csv("Gaussian_process\\X_train.csv",header=None).values
y_train = pd.read_csv("Gaussian_process\\y_train.csv",header=None).values
X_test = pd.read_csv("Gaussian_process\\X_test.csv",header=None).values
y_test = pd.read_csv("Gaussian_process\\y_test.csv",header=None).values


# In[70]:


def RMSE(y_predicted,y_test):
    y_predicted = np.array(y_predicted)
    y_test = np.array(y_test)
    return np.sqrt(np.sum((y_predicted - y_test)**2)/len(y_test))


# In[71]:


class Gaussian_process:
    def __init__(self,X_train,y_train, sigma2, b):
        self.sigma2 = sigma2
        self.b = b
        self.X_train = X_train
        self.y_train = y_train
        
    def Kernel(self,xi,xj):
        return math.exp((-1/self.b) * np.sum((xi -xj)**2))
    
    def fit(self):
        K = []
        for i in self.X_train:
            temp = []
            for j in self.X_train:
                temp.append(self.Kernel(i,j))
            K.append(temp)
        self.K = np.asarray(K)
        
    def predict(self,x0):
        temp = []
        for j in self.X_train:
            temp.append(self.Kernel(x0,j))
        KxD = np.asarray(temp)
        sigma2I = self.sigma2 * np.identity(self.K.shape[0])
        mean = (np.matmul(np.matmul(KxD,np.linalg.inv(sigma2I + self.K)), self.y_train))[0]
        variance = self.sigma2 + self.Kernel(x0,x0) + np.matmul(np.matmul(KxD,np.linalg.inv(sigma2I + self.K)), np.transpose(KxD))
        
        return [mean,variance]


# In[51]:


B = [5,7,9,11,13,15]
Sigma2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


# In[52]:


RMSES = []
y_test = [i[0] for i in y_test]
for b in B:
    temp = [] 
    for sigma2 in Sigma2:
        y_predicted = []
        process = Gaussian_process(X_train,y_train,sigma2 = sigma2, b = b)
        process.fit()
        for i in X_test:
            y_predicted.append(process.predict(i)[0])
        temp.append(RMSE(y_predicted,y_test))
    RMSES.append(temp)


# In[53]:


rmse_values = pd.DataFrame(RMSES, columns = Sigma2, index = B)


# In[55]:


rmse_values


# In[72]:


X_carweight_train = X_train[:,3]
X_carweight_test = X_test[:,3]


# In[73]:


b2 = 5
sigma22 = 2


# In[74]:


y_predicted_train = []
process = Gaussian_process(X_carweight_train,y_train,sigma2 = sigma22, b = b2)
process.fit()
for i in X_carweight_train:
    y_predicted_train.append(process.predict(i)[0])


# In[75]:


y_train = [i[0] for i in y_train]
X_carweight_train = list(X_carweight_train)
y_preds = [i for i,_ in sorted(zip(y_predicted_train,X_carweight_train), key=lambda pair: pair[1])]


# In[79]:


plt.figure(figsize=(5,5))
plt.scatter(X_carweight_train,y_train, label = "True values of y")
plt.plot(sorted(X_carweight_train),y_preds,color="red",linewidth=4.0,label = "Predicted values of y")
plt.xlabel("x[4] : Car Weight")
plt.ylabel("y : Miles Per Gallon")
plt.title("Miles Per Gallon v/s Car weight")
plt.legend()
plt.show()


# In[ ]:




