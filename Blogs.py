# -*- coding: utf-8 -*-
"""ML_Group_Task_Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uIM2akL-SIghOvmYuLh05MgCY0HTcfY5
"""

from google.colab import drive
drive.mount('/content/drive')

# Importing Libraries
import numpy as np
import pandas as pd

# Loading Training Data for BLOG.
# Considering \N and nA as NaN.
df_train=pd.read_csv('/content/drive/My Drive/Colab Notebooks/team15/Blog_train.csv',na_values=['\\N','nA'])

#To check columns with for NaN, \N, Na values
df_train.isna().sum()

#Dropping all the columns with just Na,\N or Nan values
df_train=df_train.drop(['user_id'
,'session_id'
,'document_language_provided'
,'year_published'
,'number_of_authors'
,'first_author_id'
,'num_pubs_by_first_author'
,'app_version'
,'app_lang'
,'user_os'
,'user_os_version'
,'user_java_version'
,'user_timezone']
,axis=1)

#Finding the %age of Nan values in each column
df_train.isna().mean()*100

#Dropping column with more than 50% of Nan value
df_train=df_train.drop('time_recs_viewed',axis=1)

#To find unique values in each column
for i in df_train.columns:
  print(i,":",df_train[i].unique())

#Replacing spaces from query_identifier column
df_train['query_identifier']=df_train['query_identifier'].str.replace(' ','')

#Dropping columns from Training Dataset that are Na,//N or NAN in Test Dataset
df_train=df_train.drop(['clicks'
,'rec_processing_time'
,'time_recs_recieved'
,'time_recs_displayed'
,'response_delivered'
,'number_of_recs_in_set'
,'ctr'],axis=1)

#Slicing various object/categorical columns to get only essential data
df_train['query_identifier']=df_train['query_identifier'].str[:10]
df_train['request_received']=df_train['request_received'].str[:10]
df_train['local_time_of_request']=df_train['local_time_of_request'].str[:10]
df_train['algorithm_class']=df_train['algorithm_class'].str[:3]

#Converting all the date columns to numeric
df_train['local_time_of_request']=pd.to_numeric(df_train['local_time_of_request'].str.replace('/',''))
df_train['request_received']=pd.to_numeric(df_train['request_received'].str.replace('/',''))

#Forward fill all the Nan values in all the date columns(In case of last row with Nan)
df_train['local_time_of_request']=df_train['local_time_of_request'].fillna(method='ffill')
df_train['request_received']=df_train['request_received'].fillna(method='ffill')

#Backward fill all the Nan values in all the date columns (In case of 1st row with Nan)
df_train['local_time_of_request']=df_train['local_time_of_request'].fillna(method='bfill')
df_train['request_received']=df_train['request_received'].fillna(method='bfill')

#Filling all the remaining Nan values with Unknown
df_train=df_train.fillna('Unknown')

# Separating Feature and Target in Training DataSet 
X=df_train.drop('set_clicked',axis=1)
y=df_train['set_clicked']

# Splitting Training Dataset into Test and Train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7, random_state=100)

# Installing category_encoders to import Target Encoder
!pip install category_encoders

# Importing Target Encoder
from category_encoders import TargetEncoder

# creating an object "te" for Target Encoder
te=TargetEncoder()

# Fitting Target Encoder on X_train and y_train (Training Data)
te.fit(X_train,y_train)

#Transforming X_train (Training Data)
X_train=te.transform(X_train)

#Transforming X_test (Training Data)
X_test=te.transform(X_test)

#Importing Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression

# Creating object for Logistic Regression
lr=LogisticRegression()

#Fitting Logistic Regression on X_train annd y_train (Training Data)
lr.fit(X_train,y_train)
#Predicting X_test (Training Data)
y_pred_train=lr.predict(X_test)

# Finding score for the prediction on Training Dataset
lr.score(X_test,y_test)

# Run on test

# Loading Testing Data for BLOG.
# Considering \N and nA as NaN.
df_test=pd.read_csv('/content/drive/My Drive/Colab Notebooks/team15/Blog_test.csv',na_values=['\\N','nA'])

#Dropping all the columns with just Na,\N or Nan values
df_test=df_test.drop(['user_id'
,'session_id'
,'document_language_provided'
,'year_published'
,'number_of_authors'
,'first_author_id'
,'num_pubs_by_first_author'
,'app_version'
,'app_lang'
,'user_os'
,'user_os_version'
,'user_java_version'
,'user_timezone']
,axis=1)

#Dropping column with more than 50% of Nan values
df_test=df_test.drop('time_recs_viewed',axis=1)

#Replacing spaces from query_identifier column
df_test['query_identifier']=df_test['query_identifier'].str.replace(' ','')

#Dropping columns that are Na,//N or NAN in Test data set
df_test=df_test.drop(['clicks','rec_processing_time','time_recs_recieved','time_recs_displayed','response_delivered','number_of_recs_in_set','ctr'],axis=1)

#Slicing various object columns to get only essential data
df_test['query_identifier']=df_test['query_identifier'].str[:10]
df_test['request_received']=df_test['request_received'].str[:10]
df_test['local_time_of_request']=df_test['local_time_of_request'].str[:10]
df_test['algorithm_class']=df_test['algorithm_class'].str[:3]

#Converting all the date columns to numeric
df_test['local_time_of_request']=pd.to_numeric(df_test['local_time_of_request'].str.replace('/',''))
df_test['request_received']=pd.to_numeric(df_test['request_received'].str.replace('/',''))

#Forward fill all the Nan values in all the date columns(In case of last row with Nan)
df_test['local_time_of_request']=df_test['local_time_of_request'].fillna(method='ffill')
df_test['request_received']=df_test['request_received'].fillna(method='ffill')

#Backward fill all the Nan values in all the date columns (In case of 1st row with Nan)
df_test['local_time_of_request']=df_test['local_time_of_request'].fillna(method='bfill')
df_test['request_received']=df_test['request_received'].fillna(method='bfill')

#Filling all the remmaining Nan values with Unknown
df_test=df_test.fillna('Unknown')

# Seperating the Features and Target in Test Dataset
X_test_final=df_test.drop('set_clicked',axis=1)
y_test_final=df_test['set_clicked']

# Applying Target Encoding on the entire Training Dataset excluding Target ('set_clicked')
X=te.transform(X)

# Applying Target Encoding on the entire Testing Dataset excluding Target ('set_clicked')
X_test_final=te.transform(X_test_final)

# Applying Logistic Regression on entire  Training Data set
y_pred_train=lr.fit(X,y)

# Predicting Target Data 
y_pred_test=lr.predict(X_test_final)

y_pred_test

df_test.head()

# Creating a submission Dataframe
submissions = pd.DataFrame()
# Adding column "recommendation_set_id" to submissions Dataframe
submissions['recommendation_set_id'] = df_test['recommendation_set_id'].astype(int)
# Adding column "set_clicked" to submissions Dataframe
submissions['set_clicked'] = y_pred_test

submissions.head()

#submissions.to_csv('Blog_predict_final.csv',index=False)

