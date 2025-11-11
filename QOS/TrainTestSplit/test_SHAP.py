import shap
import pandas as pd
import numpy as np
shap.initjs()

customer = pd.read_csv('C:/Users/caurel/OneDrive - Capgemini/Documents/Python/.venvTrafficLight/SUMO-Traffic-Simulator-Tutorial/QOS/TrainTestSplit/customer_churn.csv', delimiter=';')
customer.head()
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = customer.drop(columns=['Churn'], axis=1) # Independent variables
y = customer.Churn # Dependent variable

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))
explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:,:,0], X_test)
shap.summary_plot(shap_values[:,:,1], X_test)
#shap.summary_plot(shap_values, X_test)
#shap.summary_plot(shap_values[0], X_test)
#.plots.waterfall(shap_values, max_display=12)
#shap.dependence_plot("Subscription Length", shap_values[0], X_test,interaction_index="Age")







