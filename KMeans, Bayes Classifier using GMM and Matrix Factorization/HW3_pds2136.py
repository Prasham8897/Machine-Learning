#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix


# # Problem 1 (K-means)

# In[2]:


pi = [0.2,0.5,0.3]
num_obs = 500


# In[3]:


mean = np.array([[0,0],[3,0],[0,3]])
cov = np.array([[1,0],[0,1]])

data= []
label = []
for _ in range(num_obs):
    gaus_index = np.random.choice(3,p=pi)
    label.append(gaus_index)
    x,y = (np.random.multivariate_normal(mean[gaus_index], cov, 1).T)
    data.append([x[0],y[0]])
data = np.array(data)
# In[5]:


scatter = plt.scatter(data[:,0],data[:,1],c=label)
plt.scatter(mean[:,0],mean[:,1],c="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original Distribution of points")
plt.show()


# In[6]:


def K_Means(data,K,num_iter=20,plot=False,show_values=False):
    num_iter = num_iter
    num_obs = len(data)
    c = np.zeros(num_obs)
    mu =np.array(random.sample(list(data),K))
    if(show_values):
        print("Initialized cluster centers are:")
        print(mu)
    if(plot):
        plt.scatter(data[:,0],data[:,1],c=c)
        plt.scatter(mu[:,0],mu[:,1],c="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.suptitle("Distribution of points (colored by clusters)")
        plt.title("(Initially assigning to one cluster)")
        plt.show()
    objective = [] 
    for _ in range(num_iter):
        for i in range(num_obs):
            temp = [np.linalg.norm(data[i]-val)**2 for val in mu]
            c[i] = (np.argmin(temp))
        objective.append(compute_KMeans_Objective(data,c,mu))
        for i in range(len(mu)):
            temp = [data[index] for index in range(num_obs) if c[index] == i]
            mu[i] = (np.mean(temp,axis=0))
        objective.append(compute_KMeans_Objective(data,c,mu))
    if(plot):
        plt.scatter(data[:,0],data[:,1],c=c)
        plt.scatter(mu[:,0],mu[:,1],c="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Distribution of points (colored by clusters)")
        plt.show()
    if(show_values):
        print("The learned cluster centers are:")
        print(mu)
    
    return [c,mu,objective]


# In[7]:


def compute_KMeans_Objective(d,labels,centers):
    loss = 0
    for i in range(len(d)):
        for j in range(len(centers)):
            if(labels[i]==j):
                loss+=np.linalg.norm(data[i]-centers[j])**2
    return loss


# In[8]:


Ks = [2,3,4,5]
Cs = []
MUs = []
OBJs = []
for k in Ks:
    plot= k == 3 or k==5
    c,mu,obj = K_Means(data,k,num_iter=20,plot=plot)
    Cs.append(c)
    MUs.append(mu)
    OBJs.append(obj)


# In[9]:


for i in range(len(OBJs)):
    obj = OBJs[i]
    obj1 = [obj[i] for i in range(len(obj)) if i%2==0]
    obj2 = [obj[i] for i in range(len(obj)) if i%2!=0]
    plt.plot([x * .5 for x in range(1,41)],obj, color ="green")
    plt.plot([x * .5 for x in range(1,41,2)],obj1,"o",color="blue",mfc='none')
    plt.plot([x * .5 for x in range(2,41,2)],obj2,"o",color="red",mfc='none')
    plt.xticks(range(0,21))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Objective Function")
    plt.title("Value of the Objective Function for K-Means for K = " + str(Ks[i]))
    plt.show()    


# # Problem 2 (Bayes classifier revisited)

# In[3]:


X_train = pd.read_csv("Prob2_Xtrain.csv",header=None).values
X_test = pd.read_csv("Prob2_Xtest.csv",header=None).values
y_train = pd.read_csv("Prob2_ytrain.csv",header=None).values
y_test = pd.read_csv("Prob2_ytest.csv",header=None).values


# In[4]:


y_train = np.array([y_train[i][0] for i in range(len(y_train))])
y_test = np.array([y_test[i][0] for i in range(len(y_test))])


# In[5]:


X_train_0 = X_train[y_train == 0]
X_train_1 = X_train[y_train == 1]


# In[6]:


data = [X_train_0,X_train_1]


# In[7]:


def Naive_Bayes(data, pi, mu , sigma, class_priors,num_classes=2):
    y_pred = np.zeros(len(data))
    K = len(pi[0])
    for i in range(len(data)):
        prob = np.zeros(num_classes)
        class_index = range(num_classes)
        for index in class_index:
            class_cond_prob = 0
            for k in range(K):
                N = multivariate_normal.pdf(data[i],mean=mu[index][k],cov=sigma[index][k])
                class_cond_prob+=((pi[index][k])*N)
            prob[index] = class_cond_prob
        label = np.argmax(prob)
        y_pred[i] = label
    return y_pred       


# In[8]:


def EM_GMM(data,k = 3,num_iter = 30,num_run = 10,compute_objective=True):
    
    num_obs = len(data)
    Objectives = []
    best_phi = np.zeros((num_obs,k))
    best_pi = np.full((k,1),1/k)
    best_mu = np.random.multivariate_normal(np.mean(data,axis=0), np.cov(data.T), k)
    best_Sigma = [np.cov(data.T)] * k
    best_objective=-1
    
    for run in range(num_run):
        phi = np.zeros((num_obs,k))
        pi = np.full((k,1),1/k)
        mu = np.random.multivariate_normal(np.mean(data,axis=0), np.cov(data.T), k)
        Sigma = np.full((k,data[0].shape[0],data[0].shape[0]),np.cov(data.T))
        print("starting run: " + str(run))
        objective = []
        for _ in range(num_iter):
            for i in range(num_obs):
                for j in range(k):
                    phi[i][j] = (pi[j] * multivariate_normal.pdf(data[i],mean=mu[j],cov=Sigma[j],allow_singular=True))
                denominator = sum(phi[i])
                phi[i] = (phi[i]/denominator)
            nk = np.sum(phi,axis=0)
            pi = (nk/num_obs)
            numerator_mu = np.zeros((k,data[0].shape[0]))
            numerator_Sigma = np.zeros((k,data[0].shape[0],data[0].shape[0]))
            for i in range(k):
                for j in range(num_obs):
                    numerator_mu[i] += (phi[j][i] * data[i])
                mu[i] = numerator_mu[i] / nk[i]
                for j in range(num_obs):
                    temp = (data[j] - mu[i]).reshape(data[j].shape[0],1)
                    numerator_Sigma[i] += (phi[j][i] * np.matmul(temp,temp.T))
                Sigma[i] = numerator_Sigma[i] / nk[i]
            if compute_objective:
                L = 0
                log_pi = np.where(pi > np.exp(-20), np.log(pi), -20)
                for i in range(num_obs):
                    for j in range(k):
                        M = multivariate_normal.pdf(data[i],mean=mu[j],cov=Sigma[j],allow_singular=True)
                        if(M<np.exp(-20)):
                            log_M = -20
                        else:
                            log_M = np.log(M)
                        N = log_pi[j]
                        L+=(phi[i][j]*(N + log_M))
                objective.append(L)
        if compute_objective:
            print("Objective value for " + str(run) + " run is: " + str(objective[-1]))
            Objectives.append(objective)
            if(objective[-1]>=best_objective):
                best_pi=pi
                best_mu=mu
                best_Sigma=Sigma
                best_phi=phi
                best_objective=objective[-1]
            print("best objective for this run is: " + str(best_objective))
    return [Objectives,best_mu,best_pi,best_Sigma,best_phi]


# In[9]:


num_class = 2
class_priors = np.zeros(num_class)
for i in range(num_class):
    class_priors[i] = len(data[i])
class_priors /= (np.sum(class_priors))


# In[9]:


print("Starting EM for class 0")
EM0 = EM_GMM(data[0],k = 3,num_iter = 30,num_run = 10,compute_objective=True)


# In[10]:


print("Starting EM for class 1")
EM1 = EM_GMM(data[1],k = 3,num_iter = 30,num_run = 10,compute_objective=True)
EM = [EM0,EM1]


# In[12]:


for num in range(num_class):
    plt.figure(figsize=(7,7))
    for i in range(len(EM[num][0])):
        plt.plot(range(5,31),EM[num][0][i][4:],label=str(i+1))
    plt.xlabel("Number of iterations")
    plt.ylabel("Log Joint Likelihood ")
    plt.suptitle("For Class: " + str(num))
    plt.title("Log marginal objective function for a 3-Gaussian mixture model over 10 different runs and for iterations 5 to 30 ")
    plt.legend()
    plt.show()


# In[13]:


MU = np.array([EM[0][1],EM[1][1]])
PI = np.array([EM[0][2],EM[1][2]])
SIGMA = np.array([EM[0][3],EM[1][3]])

predictions = Naive_Bayes(data = X_test,
                          pi = PI,
                          mu = MU,
                          sigma = SIGMA,
                          class_priors = class_priors, 
                          num_classes = num_class)

conf_mat = confusion_matrix(y_true = y_test, y_pred = predictions)
print("The results for 3- Gaussian Mixture Model")
print(pd.DataFrame(conf_mat))
accuracy = round((conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat),2)
print("Accuracy: " + str(accuracy))


# In[10]:


K = [1,2,4]
for k in K:
    print(k)
    print("Starting EM for class 0")
    EM0 = EM_GMM(data[0],k = k,num_iter = 30,num_run = 10)
    print("Starting EM for class 1")
    EM1 = EM_GMM(data[1],k = k,num_iter = 30,num_run = 10)
    EM1 = [EM0,EM1] 
    MU = np.array([EM1[0][1],EM1[1][1]])
    PI = np.array([EM1[0][2],EM1[1][2]])
    SIGMA = np.array([EM1[0][3],EM1[1][3]])
    predictions = Naive_Bayes(data = X_test,
                              pi = PI, 
                              mu = MU, 
                              sigma = SIGMA, 
                              class_priors = class_priors, 
                              num_classes = num_class)
    conf_mat = confusion_matrix(y_true = y_test, y_pred = predictions)
    print("The results for " +str(k)+"- Gaussian Mixture Model")
    print(pd.DataFrame(conf_mat))
    accuracy = round((conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat),2)
    print("Accuracy: " + str(accuracy))


# # Problem 3 (Matrix factorization)

# In[4]:


def RMSE(y_predicted,y_test):
    return np.sqrt(np.sum((y_predicted - y_test)**2)/len(y_test))


# In[5]:


ratings_train = pd.read_csv("Prob3_ratings.csv",header=None,names=["user_id","movie_id","ratings"])
ratings_test = pd.read_csv("Prob3_ratings_test.csv",header=None,names=["user_id","movie_id","ratings"]) 


# In[6]:


list_of_movies = []
f = open("Prob3_movies.txt","r")
for line in f:
    list_of_movies.append(line.strip())


# In[8]:


sigma2 = 0.25
d = 10
lambda_val = 1
num_iter = 100
num_runs = 10


# In[9]:


SigmaUi = {}
SigmaVj = {}
user_mapping = {}
movie_mapping = {}
user_index = 0
movie_index = 0
for i in list(sorted(ratings_train["user_id"].unique())):
    user_mapping[i] = user_index
    dictui={user_index:[]}
    SigmaUi.update(dictui)
    user_index+=1
for i in list(sorted(ratings_train["movie_id"].unique())):
    movie_mapping[i] = movie_index
    dictui={movie_index:[]}
    SigmaVj.update(dictui)
    movie_index+=1


# In[10]:


num_users = len(user_mapping)
num_items = len(movie_mapping)


# In[11]:


M = ratings_train.pivot(index="user_id",columns="movie_id",values="ratings")
M.index = M.index.map(user_mapping)
M.columns = M.columns.map(movie_mapping)
M_array = np.array(M)


# In[12]:


Sigma = [tuple(pair) for pair in np.argwhere(M.notnull().values).tolist()]


# In[13]:


for i,j in Sigma:
    SigmaUi[i].append(j)
    SigmaVj[j].append(i)


# In[14]:


ratings_test["user_id"] = ratings_test["user_id"].map(user_mapping)
ratings_test["movie_id"] = ratings_test["movie_id"].map(movie_mapping)

new_test = ratings_test.dropna()
test_users_list = [int(val) for val in list(new_test["user_id"])]
test_items_list = [int(val) for val in list(new_test["movie_id"])]
y_test = new_test["ratings"].values


# In[432]:


best_log_likelihood = 100000
likelihoods = []
RMSES=[]
best_U = np.zeros([num_users,d])
best_V = np.zeros([num_items,d])
for num in range(num_runs):
    U = np.random.multivariate_normal([0]*d, lambda_val**-1 * np.identity(d), num_users)
    V = np.random.multivariate_normal([0]*d, lambda_val**-1 * np.identity(d), num_items)
    log_likelihood = []
    for _ in range(num_iter):
        u_norm = 0
        v_norm = 0
        temp = 0 
        for i in range(num_users):
            first = (lambda_val * sigma2 * np.identity(d))
            vj = V[SigmaUi[i]]
            second = np.matmul(vj.T, vj)
            first_inv = np.linalg.inv(first + second)
            Mij = M_array[i,SigmaUi[i]]
            second_term = np.matmul(vj.T,Mij)
            update = np.matmul(first_inv,second_term)
            U[i]= update
            u_norm+=np.linalg.norm(U[i])**2
        for i in range(num_items):
            first = (lambda_val * sigma2 * np.identity(d))
            ui = U[SigmaVj[i]]
            second = np.matmul(ui.T, ui)
            first_inv = np.linalg.inv(first + second)
            Mij = M_array[SigmaVj[i],i]
            second_term = np.matmul(ui.T,Mij)
            update = np.matmul(first_inv,second_term)
            V[i]= update
            v_norm+=np.linalg.norm(V[i])**2
            temp+=np.linalg.norm(Mij - np.matmul(ui,V[i].T))**2
        likelihood = -1*((temp*0.5 / sigma2)+ (-lambda_val * u_norm * 0.5) + (-lambda_val * v_norm * 0.5))
        log_likelihood.append(likelihood)
    likelihoods.append(log_likelihood)
    
    if(best_log_likelihood==100000):
        best_log_likelihood = log_likelihood[99]
    elif(log_likelihood[99]>=best_log_likelihood):
        best_log_likelihood = log_likelihood[99]
        best_U = U
        best_V = V
    print("The best log joint likelihood value till " + str(num+1)+ " run is: " + str(best_log_likelihood))
    
    u = U[test_users_list]
    v = V[test_items_list]
    z = np.multiply(u,v)
    predictions = np.sum(z,axis=1)
    rmse = RMSE(predictions,y_test)
    RMSES.append(rmse)


# In[501]:


for i in range(len(likelihoods)):
    plt.plot(likelihoods[i],label=str(i+1))
plt.xlabel("Number of iterations")
plt.ylabel("Log Joint Likelihood ")
plt.legend()
plt.show()


# In[441]:


joint_log_Likelihood = [i[-1] for i in likelihoods]
reqd_values = pd.DataFrame({"Likelihood":joint_log_Likelihood,"RMSE":RMSES})
reqd_values.sort_values("Likelihood",ascending=False)


# In[18]:


movies = ["Star Wars (1977)","My Fair Lady (1964)","GoodFellas (1990)"]
indices = [list_of_movies.index(i) for i in movies]
index_movies = [1+val for val in indices]
index_movies = list(map(lambda x:movie_mapping[x], index_movies))


# In[526]:


similar_movies = []
distances= []
for i in index_movies:
    v = best_V[i]
    temp1 = np.sort([np.linalg.norm(v- best_V[ind])**2 for ind in range(len(best_V)) if ind!=i])
    temp = np.argsort([np.linalg.norm(v- best_V[ind])**2 for ind in range(len(best_V)) if ind!=i])
    similar_movies.append(list(temp[:10]))
    distances.append(temp1[:10])


# In[527]:


inv_movie_mapping = {v: k for k, v in movie_mapping.items()}


# In[528]:


similar_movies = [list(map(lambda x:list_of_movies[inv_movie_mapping[x]-1], i)) for i in similar_movies]


# In[534]:


for i in range(len(movies)):
    print("-----------------------------------------------------------")
    print("Similar movies for " + movies[i])
    print(pd.DataFrame({"Movies":similar_movies[i],"Distances": distances[i]}))

