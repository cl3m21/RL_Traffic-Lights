# Script Name : SKLearn.py :
# Author Clement AUREL Clement.aurel@capgemini.com
# Date: November 2025
# Scope Route 25 R&D
# November technical tasks
# Description : Different models are used in this script, Linear, XForest, XGBoost , Evaluation is done, study with SHAP to find most important features for the model creation...
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt


###  START IMPORT DATASET ###
##
#

df = pd.read_csv('qws1.csv', encoding = 'latin-1')
# df.head()

#Check all columns are numerical/string but not objects and convert them if any (in this case the 2 columns found will be dropped anyway, Service Name &  WSDL Address
#df.dtype

for i in df.columns:
    if df[i].dtype == 'O':#Letter O for object not 0 (zero)
        print(df[i])
        df[i] = pd.Categorical(df[i])
        df[i] = df[i].cat.codes


#Cleaning unecessary columns
X = df.drop(columns=['Class','Service Name', 'WSDL Address'])
y = df['Class']


#
##
### END IMPORT DATASET ###

### START SPLIT DATASET ###
##
#


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

print("Training set shape : ", X_train.shape, y_train.shape)

print("Test set shape : " , X_test.shape, y_test.shape)


#
##
### END SPLIT DATASET ###

### START LINEAR REGRESSION MODEL #####
##
#

regLin = LinearRegression(fit_intercept = True)

#Train the model on training set X_train & y_train
regLin.fit(X_train, y_train)

# predict on test set with model based on features , target being the classification value
y_predLin = regLin.predict(X_test)
print("prediction values from model output Linear Regression")
print(y_predLin)

#Compare with test set results from dataset and measure performance
#put output from test Y prediction into np array to compare them (rounding the second np to 0
if np.array_equal(np.array(y_test), np.array(y_predLin).round(0)) == False:
    print("dont match");
    #count non zero so good result
    print("Percentage of match of the model vs the dataset results ", ((np.count_nonzero(np.array(y_test) == np.array(y_predLin))) / np.size(np.array(y_test) == np.array(y_predLin)) * 100).round() , "%")

else:
    # match is 100%
    print("Percentage of match of the model vs the dataset results is 100% ")


# all prediction classifications matches the values from the y_test value confirming our model works perfectly with Linear Regression
#Measuring performance on test set
print("test training over the test set")
score = regLin.score(X_test, y_test)
print(score)
print("end linear ML")
#
##
### END LINEAR REGRESSION MODEL ###


### START LOGISTIC REGRESSION MODEL ###
##
#

regLog = LogisticRegression()
regLog.fit(X_train, y_train)

y_predLog = regLog.predict(X_test)

#comparing tables

np.array_equal(np.array(y_test), np.array(y_predLog))

#getting % of diff

np.array(y_test) == np.array(y_predLog)

print("prediction values from model output Logistic Regression")
print(y_predLog)


#then compare with test set results from dataset and measure performance
#put output from test Y prediction into np array to compare them

compare = np.array(y_test) == np.array(y_predLog)

np.array_equal(np.array(y_test), np.array(y_predLog))
#dont match so model is not 100% accurate
if np.array_equal(np.array(y_test), np.array(y_predLog)) == False:
    print("dont match");
    #count non zero so good result
    print("Percentage of match of the model vs the dataset results ", ((np.count_nonzero(np.array(y_test) == np.array(y_predLog))) / np.size(np.array(y_test) == np.array(y_predLog)) * 100).round() , "%")

    np.size(np.array(y_test) == np.array(y_predLog))

print("fin")

# all prediction classifications matches the values from the y_test value confirming our model works perfectly with Linear Regression
#Measuring performance on test set

#
##
### END LOGISTIC REGRESSION MODEL ###

### START RANDOM FOREST CLASSIFIER ###
##
#

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

#Get accuract score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#COMPUTE CONFUSION MATRIX

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix  Random Forest Classifier Model")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_pred, y_test)
print("Classification Random Forest model")
print(class_report)



#
##
### END RANDOM FOREST CLASSIFIER ###


### START SHAP INTERPRETATION  ###
###
##
#

import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
shap.initjs()


explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values[:,:,1], X_test,class_inds="original", class_names=clf.classes_)

#Applying a small trick to be able to have the title in the plot, the title argument in the summary_plot doesn't work
#Also the category is in reality the index +1 so we have to make the changes when printing the output to have the good class. Index 0 is Class 1, index 1 is class 2, index 2 is class 3 ,  index 3 is class 4

#Class 1
#Bar plot
bar_plot = shap.summary_plot(shap_values[:,:,0], X_test,plot_type="bar",show=False)
plt.title('Class1_bar')
plt.savefig('Class1_bar.png')
plt.close()
#Heat point plot
point_plot = shap.summary_plot(shap_values[:,:,0], X_test,show=False)
plt.title('Class1_points')
plt.savefig('Class1_points.png')
plt.close()

#Class 2
#Bar plot
bar_plot = shap.summary_plot(shap_values[:,:,1], X_test,plot_type="bar",show=False)
plt.title('Class2_bar')
plt.savefig('Class2_bar.png')
plt.close()
#Heat point plot
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class2_points')
plt.savefig('Class2_points.png')
plt.close()

#Class 3
#Bar plot
bar_plot = shap.summary_plot(shap_values[:,:,2], X_test,plot_type="bar",show=False)
plt.title('Class3_bar')
plt.savefig('Class3_bar.png')
plt.close()
#Heat point plot
point_plot = shap.summary_plot(shap_values[:,:,2], X_test,show=False)
plt.title('Class3_points')
plt.savefig('Class3_points.png')
plt.close()

#Class 4
#Bar plot
bar_plot = shap.summary_plot(shap_values[:,:,3], X_test,plot_type="bar",show=False)
plt.title('Class4_bar')
plt.savefig('Class4_bar.png')
plt.close()
#Heat point plot
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class4_points')
plt.savefig('Class4_points.png')
plt.close()

#
##
###
#### END SHAP INTERPRETATION ####


import xgboost

import shap

# train XGBoost model
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

###PIPELINE CREATION

#from sklearn.pipeline import make_pipeline

