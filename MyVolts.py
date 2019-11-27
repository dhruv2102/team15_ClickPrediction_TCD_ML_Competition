import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics
from scipy import stats
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def S2(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:8]
    else:
        return s1

def S3(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:10]
    else:
        return s1        

def S4(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:3]
    else:
        return s1

def S5(s1):
    if (s1 is not None):
        return str(s1).replace('-','')
    else:
        return s1        

def S6(s1):
    if (s1 is not None):
        return str(s1).replace('/','')
    else:
        return s1        


dataset=pd.read_csv('MyVolts_train.csv', na_values=["\\N","nA"])
dataset1
dataset.isnull().any()
# dataset['Gender']=dataset['Gender'].fillna('missing')
# dataset['Profession']=dataset['Profession'].fillna('missing')
# dataset['University Degree']=dataset['University Degree'].fillna('missing')
# dataset['Hair Color']=dataset['Hair Color'].fillna('missing')
# dataset.isnull().any()
# dataset['Year of Record']

#dropping cols which have \N and nA in test data as they are not required for learning
datasetnoncateg=dataset.drop(['response_delivered','rec_processing_time','number_of_recs_in_set','time_recs_recieved','time_recs_displayed','time_recs_viewed','clicks','ctr','user_os','user_os_version','user_java_version','user_timezone','document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version'],axis=1)
#find the % of missing values in each col
datasetnoncateg.isnull().mean().round(4) *100
#drop cols with more than 50% of missing values
datasetnoncateg=datasetnoncateg.drop(['timezone_by_ip','local_time_of_request','local_hour_of_request'],axis=1)
datasetnoncateg.shape
# datasetnoncateg.to_csv(r'C:\projects\tcd-ml-comp-201920-rec-alg-click-pred-group\datasetnoncateg1.csv',index=False)

#imputation of cols will  be taken care by target encoder in case of cols with string values

#trimming and imputation of the cols
from sklearn.impute import SimpleImputer
simpleimputermedian=SimpleImputer(strategy='median')
datasetnoncateg['query_char_count']=simpleimputermedian.fit_transform(datasetnoncateg['query_char_count'].values.reshape(-1,1))
datasetnoncateg['query_word_count']=simpleimputermedian.fit_transform(datasetnoncateg['query_word_count'].values.reshape(-1,1))
datasetnoncateg['query_document_id']=simpleimputermedian.fit_transform(datasetnoncateg['query_document_id'].values.reshape(-1,1))
datasetnoncateg['abstract_word_count']=simpleimputermedian.fit_transform(datasetnoncateg['abstract_word_count'].values.reshape(-1,1))
datasetnoncateg['abstract_char_count']=simpleimputermedian.fit_transform(datasetnoncateg['abstract_char_count'].values.reshape(-1,1))
datasetnoncateg.query_identifier = list(datasetnoncateg.query_identifier.map(S2))
datasetnoncateg.item_type = list(datasetnoncateg.item_type.map(S2))
datasetnoncateg.request_received = list(datasetnoncateg.request_received.map(S3))
datasetnoncateg.request_received = list(datasetnoncateg.request_received.map(S5))
datasetnoncateg.request_received = list(datasetnoncateg.request_received.map(S6))
datasetnoncateg.request_received =pd.to_numeric(datasetnoncateg['request_received'])
datasetnoncateg['request_received']=datasetnoncateg['request_received'].fillna(method='ffill')
datasetnoncateg['request_received']=datasetnoncateg['request_received'].fillna(method='bfill')
datasetnoncateg.algorithm_class = list(datasetnoncateg.algorithm_class.map(S4))
datasetnoncateg.cbf_parser = list(datasetnoncateg.cbf_parser.map(S4))

# datasetnoncateg.to_csv(r'C:\projects\tcd-ml-comp-201920-rec-alg-click-pred-group\datasetnoncateg2.csv',index=False)
datasetnoncateg.isnull().any()



# dataset['Age']=simpleimputermedian.fit_transform(dataset['Age'].values.reshape(-1,1))
# dataset['Body Height [cm]']=simpleimputermedian.fit_transform(dataset['Body Height [cm]'].values.reshape(-1,1))
# datasetnoncateg.Profession = list(datasetnoncateg.Profession.map(S2))

#datasetcateg=pd.get_dummies(datasetnoncateg, prefix_sep='_')
#colsToDrop = [col for col in datasetcateg.columns if 'missing' in col]
#datasetcategnonmissing=datasetcateg.drop(colsToDrop,axis=1)



M=pd.read_csv('MyVolts_test.csv', na_values=["\\N","nA"])
# M.isnull().any()
# M['Gender']=M['Gender'].fillna('missing')
# M['Profession']=M['Profession'].fillna('missing')
# M['University Degree']=M['University Degree'].fillna('missing')
# M['Hair Color']=M['Hair Color'].fillna('missing')

#dropping cols which have \N and nA in test data as they are not required for learning
Mnoncateg=M.drop(['response_delivered','rec_processing_time','number_of_recs_in_set','time_recs_recieved','time_recs_displayed','time_recs_viewed','clicks','ctr','user_os','user_os_version','user_java_version','user_timezone','document_language_provided','year_published','number_of_authors','first_author_id','num_pubs_by_first_author','app_version'],axis=1)
#find the % of missing values in each col
Mnoncateg.isnull().mean().round(4) *100
#drop cols with more than 50% of missing values like in training
Mnoncateg=Mnoncateg.drop(['timezone_by_ip','local_time_of_request','local_hour_of_request'],axis=1)
Mnoncateg=Mnoncateg.drop(['set_clicked'],axis=1)
#trimming the cols
from sklearn.impute import SimpleImputer
simpleimputermedian=SimpleImputer(strategy='median')
Mnoncateg['query_char_count']=simpleimputermedian.fit_transform(Mnoncateg['query_char_count'].values.reshape(-1,1))
Mnoncateg['query_word_count']=simpleimputermedian.fit_transform(Mnoncateg['query_word_count'].values.reshape(-1,1))
Mnoncateg['query_document_id']=simpleimputermedian.fit_transform(Mnoncateg['query_document_id'].values.reshape(-1,1))
Mnoncateg['abstract_word_count']=simpleimputermedian.fit_transform(Mnoncateg['abstract_word_count'].values.reshape(-1,1))
Mnoncateg['abstract_char_count']=simpleimputermedian.fit_transform(Mnoncateg['abstract_char_count'].values.reshape(-1,1))
Mnoncateg.query_identifier = list(Mnoncateg.query_identifier.map(S2))
Mnoncateg.item_type = list(Mnoncateg.item_type.map(S2))
Mnoncateg.request_received = list(Mnoncateg.request_received.map(S3))
Mnoncateg.request_received = list(Mnoncateg.request_received.map(S5))
Mnoncateg.request_received = list(Mnoncateg.request_received.map(S6))
Mnoncateg.request_received =pd.to_numeric(Mnoncateg['request_received'])
Mnoncateg['request_received']=datasetnoncateg['request_received'].fillna(method='ffill')
Mnoncateg['request_received']=datasetnoncateg['request_received'].fillna(method='bfill')
Mnoncateg.algorithm_class = list(Mnoncateg.algorithm_class.map(S4))
Mnoncateg.cbf_parser = list(Mnoncateg.cbf_parser.map(S4))


# M['Year of Record']=simpleimputermedian.fit_transform(M['Year of Record'].values.reshape(-1,1))
# M['Age']=simpleimputermedian.fit_transform(M['Age'].values.reshape(-1,1))
# M['Body Height [cm]']=simpleimputermedian.fit_transform(M['Body Height [cm]'].values.reshape(-1,1))
# Mnoncateg=M.drop(['Instance','Hair Color','Wears Glasses','Hair Color','Income'],axis=1)
# Mnoncateg.Profession = list(Mnoncateg.Profession.map(S2))
# Mcateg=pd.get_dummies(Mnoncateg, prefix_sep='_')
# colsToDropM = [col for col in Mcateg.columns if 'missing' in col]
# Mcategnonmissing=Mcateg.drop(colsToDropM,axis=1)


# for column in datasetcategnonmissing:
#     if column not in Mcategnonmissing:
#         if column != 'Income in EUR':
#             Mcategnonmissing[column]=0


# for column in Mcategnonmissing:
#     if column not in datasetcategnonmissing:
#         datasetcategnonmissing[column]=0         


#datasetcategnonmissing=datasetcategnonmissing.sort_index(axis=1)
#Mcategnonmissing=Mcategnonmissing.sort_index(axis=1)

#z1 = np.abs(stats.zscore(datasetcategnonmissing))
#z2 = np.abs(stats.zscore(Mcategnonmissing))
#datasetcategnonmissing=datasetcategnonmissing[(z1 < 3).all(axis=1)]
#Mcategnonmissing=Mcategnonmissing[(z2 < 3).all(axis=1)]

#X=datasetcategnonmissing.drop('Income in EUR',axis=1).values
#Y=datasetcategnonmissing['Income in EUR'].values


X=datasetnoncateg.drop('set_clicked',axis=1).values
Y=datasetnoncateg['set_clicked'].values
#target encoding
t1 = TargetEncoder()
t1.fit(X, Y)
X = t1.transform(X)

X.isnull().any()

#auto feature selection
# K= SelectKBest(f_regression, k=10)
# K.fit(X,Y)
# X=K.transform(X)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
# regressor = BayesianRidge()
# regressor = RandomForestClassifier(n_estimators=100, min_samples_leaf=3)
#regressor = AdaBoostRegressor()
#regressor = = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
#regressor = XGBRegressor()
# regressor = GradientBoostingRegressor(n_estimators=1000)
# regressor = LogisticRegression()
# regressor=GaussianNB()
# regressor=DecisionTreeClassifier(random_state=0)
regressor = RandomForestClassifier(n_estimators=1000)



fitResult = regressor.fit(Xtrain, Ytrain)
YPredTest = regressor.predict(Xtest)
#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
# np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Ytest, YPredTest))
print('Accuracy: %d',(regressor.score(Xtest,Ytest)))
print('scores: %d',(cross_val_score(regressor, X, Y, cv=5)))


A=Mnoncateg.values
A1=t1.transform(A)
# A2 = K.transform(A1)
B=regressor.predict(A1)

df2=pd.DataFrame()
df2['recommendation_set_id']=M['recommendation_set_id']
df2['set_clicked']=B

df2.to_csv(r'C:\projects\tcd-ml-comp-201920-rec-alg-click-pred-group\outputRF1002_minleaf3.csv',index=False)
