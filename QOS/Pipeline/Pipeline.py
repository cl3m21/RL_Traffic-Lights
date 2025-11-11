#
#Original code
#https://ryanandmattdatascience.com/scikit-learn-pipelines/
#
#
import pandas as pd
import numpy as np


import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


d1 = {'Social_media_followers':[1000000, np.nan, 2000000, 1310000, 1700000, np.nan, 4100000, 1600000, 2200000, 1000000],
'Sold_out':[1,0,0,1,0,0,0,1,0,1]}
df1 = pd.DataFrame(data=d1)
X1 = df1[['Social_media_followers']]
y1 = df1[['Sold_out']]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.3,random_state=19)
imputer = SimpleImputer(strategy='mean')
lr = LogisticRegression()
pipe1 = make_pipeline(imputer, lr)
pipe1.fit(X1_train, y1_train)
pipe1.score(X1_train,y1_train)
pipe1.score(X1_test,y1_test)
pipe1.named_steps.simpleimputer.statistics_
pipe1.named_steps.logisticregression.coef_

d2 = {'Genre':['Rock', 'Metal', 'Bluegrass', 'Rock', np.nan, 'Rock', 'Rock', np.nan, 'Bluegrass', 'Rock'],
'Social_media_followers':[1000000, np.nan, 2000000, 1310000, 1700000, np.nan, 4100000, 1600000, 2200000, 1000000],
'Sold_out':[1,0,0,1,0,0,0,1,0,1]}
df = pd.DataFrame(data=d2)
df.head(10)
X = df.iloc[:, 0:2]
y = df.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=75)
num_cols = ['Social_media_followers']
cat_cols = ['Genre']
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot',OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
col_trans = ColumnTransformer(transformers=[
('num_pipeline',num_pipeline,num_cols),
('cat_pipeline',cat_pipeline,cat_cols)
],
remainder='drop',
n_jobs=-1)
dtc = DecisionTreeClassifier()
pipefinal = make_pipeline(col_trans, dtc)
pipefinal.fit(X_train, y_train)
pipefinal.score(X_test, y_test)

#Save pipeline

joblib.dump(pipefinal,"pipe.joblib")
pipefinal2 = joblib.load("pipe.joblib")