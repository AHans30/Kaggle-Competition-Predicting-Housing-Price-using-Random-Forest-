# Kaggle-Competition-Predicting-Housing-Price-using-Random-Forest-

Code for Kaggle competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

After One-Hot encoding categorical variables, I use ProcessData function to add polynomial features and scale them in uniform distribution using Sklearn's StandardScaler class

Following processing of data, I use RandomForestRegressor and fit it to training data and then make predictions on test data.

I've also applied k-fold cross validation to calculate accuracy and standard deviation on test set predictions

Following, I've tuned my Random forest hyperparameters using GridSearchCV

At the satisfactory mean squared logarithim error of .14, I've learned re-learned parameters to make predictions but used whole training
data provided by 
