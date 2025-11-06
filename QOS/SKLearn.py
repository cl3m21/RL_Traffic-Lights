import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing dataset
df = pd.read_csv('qws1.csv', encoding = 'latin-1')
# df.head()

#Cleaning unecessary columns
X = df.drop(columns=['Service Name', 'WSDL Address'])
y = df['Class']

#to have a description of the table (averages, min max...)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

print("Training set shape : ", X_train.shape, y_train.shape)

print("Test set shape : " , X_test.shape, y_test.shape)

#Linear Regression model
reg = LinearRegression(fit_intercept = True)

#Train the model on training set X_train & y_train


reg.fit(X_train, y_train)

# predict on test set with model based on feature , target being the classification value

y_pred = reg.predict(X_test)
print("prediction values from model output")
print(y_pred)

#then compare with test set results from dataset and measure performance
#put output from test Y prediction into np array to compare them

np.array(y_test) == np.array(y_test)

# all prediction classifications matches the values from the y_test value confirming our model works perfectly with Linear Regression
#Measuring performance on test set

print("test training over the test set")
score = reg.score(X_test, y_test)
print(score)

#pipeline


#from sklearn.pipeline import make_pipeline