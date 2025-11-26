# Script Name : SKLearn.py :
# Author Clement AUREL Clement.aurel@capgemini.com
# Date: November 2025
# Scope Route 25 R&D
# November technical tasks
# Description : Different models are used in this script, Linear (doesn"t fit for the choosen mode of classification),Logistics, XForest, XGBoost , Evaluation is done, study with SHAP(for RandomForest and XBOOST Only) to find most important features for the model creation...
#- Confusion matrix heatmap & SHAP plots are saved in current folder, as well as  classification reports for each model.
#- Input data was cleaned from unecessary column, split in X (removed Class column but all other columns are set here) and Y (target Class only).
# -Data was scaled with StandardScale and accuracy on Logistic Regression, Model has climbed from 69.8 to 95.89.
# -Pipeline is used to concat transformation and also to export the full pipeline that includes Scaler and Model

import joblib
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




''''
#Scale data MOVED TO the pipeline
print(X)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print(X)
'''
#
##
### END IMPORT DATASET ###

#PREPROCESSING after getting this error "ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3], got [1 2 3 4]" on XGBoost Model , might as well apply for all

from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()

y = le.fit_transform(y)

#diplay mapping

mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
for k,v in mapping.items():
    print(f" Class {k}: Index {v}")


#
##
### END PREPROCESSING DATA ###



### START SPLIT DATASET ###
##
#


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

### PREPROCESSING DATA ###
##
#

print("Training set shape : ", X_train.shape)

print("Test set shape : " , X_test.shape)

#
##
### END SPLIT DATASET ###


### START LOGISTIC REGRESSION MODEL ###
##
#

### Pipeline ###
from sklearn.pipeline import make_pipeline, Pipeline
scaler = StandardScaler()
regLog =  LogisticRegression()

pipeline = make_pipeline(scaler, regLog)
pipeline.fit(X_train, y_train)

print("Score for Pipeline (Scaling StandardScaler & model logistic Regression ) : ",  (pipeline.score(X_test, y_test)*100).__round__(3) , " %" )
pipeline.named_steps.logisticregression.coef_
y_predLog = pipeline.predict(X_test)


# START COMPUTE CONFUSION MATRIX

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report

conf_matrix = confusion_matrix(y_test, y_predLog)
print("Confusion Matrix Logistic Regression Model")
print(conf_matrix)

dfCM = pd.DataFrame(conf_matrix).transpose()
#Saving result to csv
dfCM.to_csv('LogisticRegressionClassificationConfusionMatrix.csv')

#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_LogisticRegression')
plt.savefig('Confusion_Matrix_Heatmap_LogisticRegression.png')
plt.close()



# Classification Report
class_report = classification_report(y_predLog, y_test, output_dict=True)
print("Classification Logistic Regression model : ")
print(class_report)
dfCR = pd.DataFrame(class_report).transpose()

#Saving result to csv
dfCR.to_csv('LogisticRegressionClassificationReport.csv')

print("pause")

### SAVE MODEL WITH Joblib ###
import joblib

filename = 'LogisticRegressionFullPipeline.joblib'
joblib.dump(pipeline,open(filename, 'wb'))

#LOAD MODEL WITH JOBLIB ###
loaded_model = joblib.load(open(filename, 'rb'))


#TEST PREDICTION WITH IMPORTED MODEL
y_pred_LM = loaded_model.predict(X_test)

#Check accuracy score matched the exported one
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_LM)

print("Score for Logistic Regression model after export / import ) : ",  (accuracy*100).__round__(3) , " %" )


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


### Pipeline ###
from sklearn.pipeline import make_pipeline, Pipeline

scaler = StandardScaler()
clf =  RandomForestClassifier()

pipeline = make_pipeline(scaler, clf)

pipeline.fit(X_train, y_train)

print("Score for Pipeline (Scaling StandardScaler & model RandomForest ) : ",  (pipeline.score(X_test, y_test)*100).__round__(3) , " %" )

# Make prediction on the testing data
y_pred = pipeline.predict(X_test)

#Get accuracy score
score = accuracy_score(y_test, y_pred)

print ("Random Forest Classification model score : ", (score * 100).__round__(3) , "%")

#COMPUTE CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix  Random Forest Classifier Model : ")
print(conf_matrix)

dfCM = pd.DataFrame(conf_matrix).transpose()
#Saving result to csv
dfCR.to_csv('RandomForestClassificationConfusionMatrix.csv')

#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_RandomForest')
plt.savefig('Confusion_Matrix_Heatmap_RandomForest.png')
plt.close()

# Classification Report
class_report = classification_report(y_pred, y_test , output_dict=True)
print("Classification Random Forest model : ")
print(class_report)

dfCR = pd.DataFrame(class_report).transpose()

#Saving result to csv
dfCR.to_csv('RandomForestClassificationReport.csv')

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

### SAVE MODEL WITH Joblib ###
import joblib

filename = 'RandomForestClassificationFullpipeline.joblib'
joblib.dump(pipeline,open(filename, 'wb'))

#LOAD MODEL WITH JOBLIB ###
loaded_model = joblib.load(open(filename, 'rb'))


#TEST PREDICTION WITH IMPORTED MODEL
y_pred_LM = loaded_model.predict(X_test)

#Check accuracy score matched the exported one
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_LM)

print("Score for Random Forest model after export / import ) : ",  (accuracy*100).__round__(3) , " %" )

#
##
### END RANDOM FOREST CLASSIFIER ###



### XGBOOST MODEL ###
##
#

import xgboost
import shap

### Pipeline ###
from sklearn.pipeline import make_pipeline, Pipeline

scaler = StandardScaler()
clfXG =  xgboost.XGBClassifier()

pipeline = make_pipeline(scaler, clfXG)

# Fit XGBoost model

pipeline.fit(X_train, y_train)
#Predicting the Test set resuls

y_pred = pipeline.predict(X_test)

#Evaluate the classifier on the test data

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy XGBoost Model : ", (accuracy * 100.0 ).__round__(3), "%")


# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix

#COMPUTE CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix  XGBoost Model : ")
print(conf_matrix)

dfCM = pd.DataFrame(conf_matrix).transpose()
#Saving result to csv
dfCR.to_csv('XGBoostConfusionMatrix.csv')


#Save heatmap Confusion Matrix
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion_Matrix_XGBoost')
plt.savefig('Confusion_Matrix_Heatmap_XGBoost.png')
plt.close()


# Classification Report
class_report = classification_report(y_pred, y_test, output_dict=True)
print("Classification XGBoost model : ")
print(class_report)

dfCR = pd.DataFrame(class_report).transpose()

#Saving result to csv
dfCR.to_csv('XGBoostReport.csv')

#START SHAP INTERPRETATION  ###

import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
shap.initjs()


explainer = shap.Explainer(clfXG)
shap_values = explainer.shap_values(X_test)

""""
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



### SAVE MODEL WITH Joblib ###
import joblib

filename = 'XGBOOSTFullpipeline.joblib'
joblib.dump(pipeline,open(filename, 'wb'))

#LOAD MODEL WITH JOBLIB ###
loaded_model = joblib.load(open(filename, 'rb'))


#TEST PREDICTION WITH IMPORTED MODEL
y_pred_LM = loaded_model.predict(X_test)

#Check accuracy score matched the exported one
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_LM)

print("Score for XBoost model after export / import ) : ",  (accuracy*100).__round__(3) , " %" )

#
##
### END XGBOOST ###
"""" "