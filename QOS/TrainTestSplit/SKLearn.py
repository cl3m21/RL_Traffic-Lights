# Script Name : SKLearn.py :
# Author Clement AUREL Clement.aurel@capgemini.com
# Date: November 2025
# Scope Route 25 R&D
# November technical tasks
# Description : Different models are used in this script, Linear (doesn"t fit for the choosen mode of classification),Logistics, XForest, XGBoost , Evaluation is done, study with SHAP to find most important features for the model creation...
#- Confusion matrix heatmap & SHAP plots are saved in current folder, as well as  classification reports for each model.
#- Input data was cleaned from unecessary column, split in X (removed Class column but all other columns are set here) and Y (target Class only).
# -Data was scaled with StandardScale and accuracy on Logistic Regressop, Model has climbed from 69.8 to 95.89

import pandas as pd
import numpy as np
from numba.core.ir import Print
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import xgboost
import seaborn as sns


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
#dropping target data into feature data variable and extra unecessary colunms
X = df.drop(columns=['Class','Service Name', 'WSDL Address'],axis=1)
y = df['Class']

#Scale data
print(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print(X)

#
##
### END IMPORT DATASET ###




### START SPLIT DATASET ###
##
#


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

### PREPROCESSING DATA ###
##
#

#PREPROCESSING after getting this error "ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3], got [1 2 3 4]" on XGBoost Model , might as well apply for all

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

y_test_encoded = le.fit_transform(y_test)


#
##
### END PREPROCESSING DATA ###

print("Training set shape : ", X_train.shape, y_train_encoded.shape)

print("Test set shape : " , X_test.shape, y_test_encoded.shape)



#
##
### END SPLIT DATASET ###

""""
### START LINEAR REGRESSION MODEL #####
##
#


#This model doesn't fit our needs since we are looking into classification, this model will return non integer numbers and must be used for other purposes (finding exact values instead of classification)


regLin = LinearRegression(fit_intercept = True)

#Train the model on training set X_train & y_train
regLin.fit(X_train, y_train)

# predict on test set with model based on features , target being the classification value
y_predLin = regLin.predict(X_test)
print("prediction values from model output Linear Regression")
print(y_predLin)

#Compare with test set results from dataset and measure performance
#put output from test Y prediction into np array to compare them (rounding the second np to 0
if np.array_equal(np.array(y_test_encoded), np.array(y_predLin).round(0)) == False:
    print("dont match");
    #count non zero so good result
    print("Percentage of match of the model vs the dataset results ", ((np.count_nonzero(np.array(y_test_encoded) == np.array(y_predLin))) / np.size(np.array(y_test_encoded) == np.array(y_predLin)) * 100).round() , "%")

else:
    # match is 100%
    print("Percentage of match of the model vs the dataset results is 100% ")


# all prediction classifications matches the values from the y_test value confirming our model works perfectly with Linear Regression
#Measuring performance on test set
print("test training over the test set")
score = regLin.score(X_test, y_test_encoded)
print(score)
print("end linear ML")
#
##
### END LINEAR REGRESSION MODEL ###

"""""

### START LOGISTIC REGRESSION MODEL ###
##
#

regLog = LogisticRegression()
regLog.fit(X_train, y_train_encoded)

y_predLog = regLog.predict(X_test)

#Get SCORE

score = regLog.score(X_test, y_test_encoded)

print ("Logistic Regression model score : ", (score * 100).__round__(3) , "%")


""""
#comparing tables

np.array_equal(np.array(y_test_encoded), np.array(y_predLog))

#getting % of diff

np.array(y_test_encoded) == np.array(y_predLog)

print("prediction values from model output Logistic Regression")
print(y_predLog)


#then compare with test set results from dataset and measure performance
#put output from test Y prediction into np array to compare them

compare = np.array(y_test_encoded) == np.array(y_predLog)

np.array_equal(np.array(y_test_encoded), np.array(y_predLog))
#dont match so model is not 100% accurate
if np.array_equal(np.array(y_test_encoded), np.array(y_predLog)) == False:
    print("dont match");
    #count non zero so good result
    print("Percentage of match of the model vs the dataset results ", ((np.count_nonzero(np.array(y_test_encoded) == np.array(y_predLog))) / np.size(np.array(y_test_encoded) == np.array(y_predLog)) * 100).round() , "%")

    np.size(np.array(y_test_encoded) == np.array(y_predLog))

print("fin")

"""

# START COMPUTE CONFUSION MATRIX

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report

conf_matrix = confusion_matrix(y_test_encoded, y_predLog)
print("Confusion Matrix Logistic Regression Model")
print(conf_matrix)

#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_LogisticRegression')
plt.savefig('Confusion_Matrix_Heatmap_LogisticRegression.png')
plt.close()

# Classification Report
class_report = classification_report(y_predLog, y_test_encoded, output_dict=True)
print("Classification Logistic Regression model : ")
print(class_report)
dfCR = pd.DataFrame(class_report).transpose()

#Saving result to excel
dfCR.to_excel('LogisticRegressionClassificationReport.xlsx', sheet_name = 'Results')

print("pause")



#START SHAP INTERPRETATION  ###

#No  SHAP interpretation for Linear model

#### END SHAP INTERPRETATION ####

#
##
### END LOGISTIC REGRESSION MODEL ###

### START RANDOM FOREST CLASSIFIER ###
##
#

# Data Processing
import pandas as pd
import numpy as np

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

clf = RandomForestClassifier()
clf.fit(X_train, y_train_encoded)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

#Get accuracy score
score = accuracy_score(y_test_encoded, y_pred)

print ("Random Forest Classification model score : ", (score * 100).__round__(3) , "%")

#COMPUTE CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix  Random Forest Classifier Model : ")
print(conf_matrix)

#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_RandomForest')
plt.savefig('Confusion_Matrix_Heatmap_RandomForest.png')
plt.close()

# Classification Report
class_report = classification_report(y_pred, y_test_encoded , output_dict=True)
print("Classification Random Forest model : ")
print(class_report)

dfCR = pd.DataFrame(class_report).transpose()

#Saving result to excel
dfCR.to_excel('RandomForestClassificationReport.xlsx', sheet_name = 'Results')

#START SHAP INTERPRETATION  ###

import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
shap.initjs()


explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

#Applying a small trick to be able to have the title in the plot, the title argument in the summary_plot doesn't work
#Also the category is in reality the index +1 so we have to make the changes when printing the output to have the good class. Index 0 is Class 1, index 1 is class 2, index 2 is class 3 ,  index 3 is class 4

#CREATE PLOTS

#CLASS 1
#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,0], X_test,plot_type="bar",show=False)
plt.title('Class1_bar_RandomForest')
plt.savefig('Class1_bar_RandomForest.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,0], X_test,show=False)
plt.title('Class1_points_RandomForest')
plt.savefig('Class1_points_RandomForest.png')
plt.close()

#CLASS 2

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,1], X_test,plot_type="bar",show=False)
plt.title('Class2_bar_RandomForest')
plt.savefig('Class2_bar_RandomForest.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class2_points_RandomForest')
plt.savefig('Class2_points_RandomForest.png')
plt.close()

#CLASS 3

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,2], X_test,plot_type="bar",show=False)
plt.title('Class3_bar_RandomForest')
plt.savefig('Class3_bar_RandomForest.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,2], X_test,show=False)
plt.title('Class3_points_RandomForest')
plt.savefig('Class3_points_RandomForest.png')
plt.close()

#CLASS 4

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,3], X_test,plot_type="bar",show=False)
plt.title('Class4_bar_RandomForest')
plt.savefig('Class4_bar_RandomForest.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class4_points_RandomForest')
plt.savefig('Class4_points_RandomForest.png')
plt.close()


#### END SHAP INTERPRETATION ####

#
##
### END RANDOM FOREST CLASSIFIER ###



### XGBOOST MODEL ###
##
#

import xgboost
import shap


# Fit XGBoost model
model = xgboost.XGBClassifier()
model.fit(X_train, y_train_encoded)

#Predicting the Test set resuls

y_pred = model.predict(X_test)

#Evaluate the classifier on the test data

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_encoded, y_pred)

print("Accuracy XGBoost Model : ", (accuracy * 100.0 ).__round__(3), "%")


# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix

#COMPUTE CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix  XGBoost Model : ")
print(conf_matrix)

#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_XGBoost')
plt.savefig('Confusion_Matrix_Heatmap_XGBoost.png')
plt.close()


# Classification Report
class_report = classification_report(y_pred, y_test_encoded, output_dict=True)
print("Classification XGBoost model : ")
print(class_report)

dfCR = pd.DataFrame(class_report).transpose()

#Saving result to excel
dfCR.to_excel('XGBoostReport.xlsx', sheet_name = 'Results',)

#START SHAP INTERPRETATION  ###

import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
shap.initjs()


explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

#Applying a small trick to be able to have the title in the plot, the title argument in the summary_plot doesn't work
#Also the category is in reality the index +1 so we have to make the changes when printing the output to have the good class. Index 0 is Class 1, index 1 is class 2, index 2 is class 3 ,  index 3 is class 4

#CREATE PLOTS

#CLASS 1
#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,0], X_test,plot_type="bar",show=False)
plt.title('Class1_bar_XGBoost')
plt.savefig('Class1_bar_XGBoost.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,0], X_test,show=False)
plt.title('Class1_points_XGBoost')
plt.savefig('Class1_points_XGBoost.png')
plt.close()

#CLASS 2

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,1], X_test,plot_type="bar",show=False)
plt.title('Class2_bar_XGBoost')
plt.savefig('Class2_bar_XGBoost.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class2_points_XGBoost')
plt.savefig('Class2_points_XGBoost.png')
plt.close()

#CLASS 3

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,2], X_test,plot_type="bar",show=False)
plt.title('Class3_bar_XGBoost')
plt.savefig('Class3_bar_XGBoost.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,2], X_test,show=False)
plt.title('Class3_points_XGBoost')
plt.savefig('Class3_points_XGBoost.png')
plt.close()

#CLASS 4

#BAR PLOT
bar_plot = shap.summary_plot(shap_values[:,:,3], X_test,plot_type="bar",show=False)
plt.title('Class4_bar_XGBoost')
plt.savefig('Class4_bar_XGBoost.png')
plt.close()

#HEAT POINT PLOT
point_plot = shap.summary_plot(shap_values[:,:,1], X_test,show=False)
plt.title('Class4_points_XGBoost')
plt.savefig('Class4_points_XGBoost.png')
plt.close()


#### END SHAP INTERPRETATION ####


###PIPELINE CREATION

#from sklearn.pipeline import make_pipeline

