#-----------Housing Price Regression----------

import numpy as np
import pandas as pd
import matplotlib as plt
#import dataset 
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')


"""Now since to use pd.get_dummies our df will have different columns since it has different
categories we combine / concatenate both dfs to make a master df. We then continue to use
pd.get_dummies in this df but  by definition this cmd will shuffle our columns in alphabetical
order. So we preserve our inteded column names in original order use ColNames"""

ColNames = list(dataset_train.columns.values)

dataset_master = pd.concat([dataset_train, dataset_test])
#Reshuffle columns in original format
dataset_master = dataset_master[ColNames]

#list of indices of categorical variables
CategoricalColNums = [2,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,27,28,29,30,31,32,33,35,39,40,41,42,53,55,57,58,60,63,64,65,72,73,74,78,79]

A = []
for i,j in enumerate(CategoricalColNums):
    A.append(ColNames[j])

#Here A has the list of type str of names of Categorical column names

dataset_master = pd.get_dummies(dataset_master, columns = A, drop_first = True)

#Again as using pd.get_dummies our data is reshuffled in different order
#We extract the index of our dependent variable 
A = list(dataset_master.columns.values)
[i for i, j in enumerate(A) if j == 'SalePrice']

#We shift our master df's dependent variable to last column and remove Id column
A.remove('SalePrice')
A.remove('Id')
A.append('SalePrice')
dataset_master = dataset_master[A]


dataset_train = dataset_master.iloc[:1460, :]
dataset_test = dataset_master.iloc[1460:, :]

#remove depedent variable SalePrice from testset containing null values
A.remove('SalePrice')
dataset_test = dataset_test[A]

#Filling NA values with column mean
dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())

#Obtaining Learning and Target Data
X = dataset_train.iloc[:,:-1].values
y = dataset_train.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    
#Process Function    
def ProcessData(TrainMatrix, TestMatrix=None):
    
    #TrainMatrix = X_train
    #TestMatrix = X_test

    #Add features like x**2, x**3, np.exp, no.log and then do PCA
    FeaturesEnggInd = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,20,22,24,26,27,28,29,30,31,32,33,34,35]
    
    for i,j in enumerate (FeaturesEnggInd):
        TrainMatrix = np.c_[TrainMatrix, TrainMatrix[:,j]**2, np.power(TrainMatrix[:,j], 0.5), TrainMatrix[:,j]**3]
        TestMatrix = np.c_[TestMatrix, TestMatrix[:,j]**2, np.power(TestMatrix[:,j], 0.5), TestMatrix[:,j]**3]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    TrainMatrix = sc.fit_transform(TrainMatrix)
    TestMatrix = sc.transform(TestMatrix)
    

    return (TrainMatrix, TestMatrix)

#ProcessData features scales the data after adding polynomial features 
# variance of data and returns X_train, X_test
    
X_train, X_test = ProcessData(TrainMatrix = X_train,
                                                      TestMatrix = X_test)


from sklearn.ensemble import RandomForestRegressor as RFR
regressor1 = RFR (n_estimators = 500, min_samples_leaf = 1, max_depth = 20, 
                  min_samples_split = 3)
regressor1.fit(X_train, y_train)

y_pred1 = regressor1.predict(X_test)

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

Err1 = rmsle(real = y_test, predicted = y_pred1)


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor1, X = X_train, y = y_train, cv=10, n_jobs = -1)
MeanAcc = accuracies.mean()*100
MeanVar = accuracies.std()*100


#Applying GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': [7,10,20,30],
              'min_samples_split': [2,3,4], 'min_samples_leaf': [1,0.5,0.4,2]}
gs = GridSearchCV(estimator = regressor1, param_grid = parameters, 
                  scoring = 'r2', cv = 10, n_jobs = -1)
gs = gs.fit(X_train, y_train)
best_acc = gs.best_score_
best_param = gs.best_params_

#Using whole train data to submit reusults on reaching satisfactory score
X_learn = ProcessData(TrainMatrix = X)
y_learn = y
regressor_final = RFR (n_estimators = 500, min_samples_leaf = 1, max_depth = 20, 
                  min_samples_split = 3)
regressor_final.fit(X_learn, y_learn)

test_pred = regressor_final.predict(dataset_test.iloc[:,:].values)

np.savetxt("SubmitPredictions.csv", test_pred, delimiter = ',')




    



