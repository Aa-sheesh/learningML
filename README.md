# Part-1: Data Preprocessing
## Machine Learning workflow
![machine learning workflow](mlw.png)

## Training and Test Split in Model Evaluation
- Training Set : Used to build the model
- Test Set : Used to check predicted vs actual values

- Dependent Variable(y): The one column that is dependent on others. Usually the one that needs to be predicted.
- Independent Variable(x): The column that is independent.

## Feature Scaling
- Always applied to columns.
- There are two types of feature scaling:
    - Normalization:
        1. X'=(X-Xmin)/(Xmax-Xmin)
        2. X' -> [0,1]
    - Standardization
        1. X'=(X-μ)/σ
        2. x' -> [-3,3]

# Part-2:Regression
Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

In this part, you will understand and learn how to implement the following Machine Learning Regression models:

    1. Simple Linear Regression
    2. Multiple Linear Regression
    3. Polynomial Regression
    4. Support Vector for Regression (SVR)
    5. Decision Tree Regression
    6. Random Forest Regression

## 1. Simple Linear Regression

### Understanding the equation
![simple linear regression equation](slr.png)

### Understanding Ordinary Least Squares Regression
![ordinary least squarest](ordleastsquares.png)

## 2. Multiple Linear Regression
NOTE : We dont need to perform feature scaling in mlr as it is autobalanced itself.

### Understanding the equation
![MLR Equation](mlr.png)
### Understanding Linear Regression Assumptions: Linearity, Homoscedasticity
![assumptions of lr](image.png)

### How to handle categorical data in Linear Regression?
![dummy variables](image-1.png) 

### Multi co linearity in linear regression
![multi co linearity of dummy variables](image-2.png)

### Building Robust Multiple Regression Models
NOTE: 
- Backward Elimination is irrelevant in Python, because the Scikit-Learn library automatically takes care of selecting the statistically significant features when training the model to make accurate predictions.
- However, if you do really want to learn how to manually implement Backward Elimination in Python and identify the most statistically significant features, please find in this link below some old videos I made on how to implement Backward Elimination in Python:
(click here)[https://www.dropbox.com/sh/pknk0g9yu4z06u7/AADSTzieYEMfs1HHxKHt9j1ba?dl=0]

- These are old videos made on Spyder but the dataset and the code are the same as in the previous video lectures of this section on Multiple Linear Regression, except that I had manually removed the first column to avoid the Dummy Variable Trap with this line of code:
```
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
```
- Just keep this for this Backward Elimination implementation, but keep in mind that in general you don't have to remove manually a dummy variable column because Scikit-Learn takes care of it.

- And also, please find the whole code implementing this Backward Elimination technique:
```
# Multiple Linear Regression
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
 
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
 
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
 
# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(np.float64)regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
Once again, this is totally optional.
```

5 methods of building models:
![methods of building models](image-3.png)
- All in
    ![all in](image-4.png)
- Backward elimination
    ![backward elim](image-5.png)
- Forward Selevtion
    ![forward selection](image-6.png)
- Bidirectional Elimination
    ![bidirectional elim](image-7.png)
- All possible models
    ![all possible models](image-8.png)

#### Link for course slides 
Link for [course slides](https://online.fliphtml5.com/grdgl/hfrm/#p=12) here.
Get your datasets, codes, and slides [here](https://www.superdatascience.com/machine-learning)