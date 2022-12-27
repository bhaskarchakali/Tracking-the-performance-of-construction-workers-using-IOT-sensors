

## Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




### here Read the Dataset for Analysis The Productivity of labour 
data = pd.read_excel(r"C:\Users\Home\Desktop\INTERNSHIP\PROJECT2\construction dataset.xlsx")
# To know the information about data
data.info()
#### let's check the shape of the Dataset
data.shape
## Will check availability of columns in dataset
data.columns
### Let's check the Head of the Dataset
data.head()
### Will Check if there is any missing-values present in the dataset
data.isnull().sum()

### Let's Check the Age present in this Dataset
data['Age'].value_counts()
### With the Help of Value-counts we can see that at the age of 25 there are 72 Workers are working ,at the age of 35 there are 60 Workers are Working, and  
### at the age of 18 there are 6 workers are working

### Let's Check the Type_of_workmen present in this Dataset
data['Type_of_workmen'].value_counts()
## By observing this, we have 429 skilled workmen in site

### Let's Check the Scope of work in this Dataset
data['Scope of work'].value_counts()
### With the Help of Value-counts we can see that the welding Works are more when compared to other works

### Let's Check the Designation of workmen in this Dataset
data['Designation'].value_counts()
### With the Help of Value-counts we can see that the welders are more when compared to other workmen

### Let's Check the Pulse_rate details in this Dataset
data['Pulse_rate'].value_counts()

data['Bod y_Mass_ Index'].value_counts()

data['Body_temperature'].value_counts()
data['Work _area'].value_counts()
data['Scope of work'].value_counts()
data['Total_hours_worked'].value_counts()
data['Real Time Information'].value_counts()
data['Acutual Work Done(cum)'].value_counts()
data['Targeted work(cum)'].value_counts()
data['Pending_work(cum)'].value_counts()
data['Daily _Wage'].value_counts()
data['Site'].value_counts()
### Let's Check the Productivity present in this Dataset
data['Productivity'].value_counts()
## We observed that 289 workmen observed as good performers & 235 workmen observed as excellent performers

## Will check the duplicates are there in data
duplicate = data. duplicated()
sum(duplicate)

#### To know the stats about the dataset
a = data.describe()

## will check zero variance in data set
data. var()
data. var()==0

data.corr()
data.columns
### drop the columns 
data.drop(['Day','Name_ of_worker','ID_Number','Native_Place','IOT Device Name','Entry _time','Exit _time'], axis = 1, inplace = True)
data.dtypes

# Label Encoder, will change catagorical data into numercal data by using Label encoding
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for i in (1,2,6,7,9,14):
    data.iloc[:,i] = enc.fit_transform(data.iloc[:,i])
data.head() 


data.dtypes

## will change position of columns in dataset
data = data. iloc[:,[15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
data.head()

data1 = data. iloc[:,1:]
#boxplot for every columns
data1.boxplot(column=['Age', 'Type_of_workmen','Designation','Pulse_rate', 'Body_Mass_Index', 'Body_temperature','Work_area','Scope_of_work', 'Total_hours_worked',
                                         'Real_Time_Information' ,'Actual_work','Targeted_work','Pending_work','Daily_wage', 'Site'])

for column in data1:
    plt.figure()
    data1.boxplot([column])
     
# Outliers observed in some of the columns
##  Winsorization ##    will use this technique  
# pip install feature_engine    install the package
#  conda install -c conda-forge feature_engine
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Pulse_rate'])
data['Pulse_rate'] = winsor.fit_transform(data[['Pulse_rate']]) 
sns.boxplot(data['Pulse_rate'])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Real_Time_Information'])
data['Real_Time_Information'] = winsor.fit_transform(data[['Real_Time_Information']]) 
sns.boxplot(data['Real_Time_Information'])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Body_Mass_Index'])
data['Body_Mass_Index'] = winsor.fit_transform(data[['Body_Mass_Index']]) 
sns.boxplot(data['Body_Mass_Index'])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total_hours_worked'])
data['Total_hours_worked'] = winsor.fit_transform(data[['Total_hours_worked']]) 
sns.boxplot(data['Total_hours_worked'])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Pending_work'])
data['Pending_work'] = winsor.fit_transform(data[['Pending_work']]) 
sns.boxplot(data['Pending_work'])

#Q-Q plot
from scipy import stats
import pylab
stats. probplot(data.Age, dist = "norm", plot = pylab)

sns.catplot(data=data, x="Age", y="Productivity", kind="box")

sns. jointplot(x = data['Age'], y =data['Daily_wage'])

plt. figure(1, figsize = (16,10))
sns. countplot(data["Productivity"])

sns.pairplot(data)

data.dtypes

data['Productivity'] = enc.fit_transform(data['Productivity'])
data.drop(['Age'], axis = 1, inplace = True)
## Normalization function ##
def norm_func(i):
    x = (i - i.min()) / (i. max() - i. min())
    return(x)

data_norm = norm_func(data. iloc[:,1:])
    
data_norm. describe()
data_norm.dtypes

#will check clustering and observe the gropus in dataset
## for creating dendrogram 
from scipy.cluster.hierarchy import linkage, dendrogram
# import scipy.cluster.hierarchy as sch

z = linkage(data_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(100, 8));plt.title('Constructiondata Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(data_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data['clust'] = cluster_labels

data =data.iloc[:,[15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
data.head()
# Aggregate mean of each cluster
data.iloc[:, 1:].groupby(data.clust).mean()
data.iloc[:, 1:].groupby(data.clust).std()

data['clust'].value_counts()
# clust0 = 228, clust1 = 344 , clust2 = 40,
##after segmented into three clusters we Observed that in standard deviation, productivity rate is high
## in clust 0 as 0.79, next in clust2 as 0.71 and then in clust1 as 0.59
## we Observed that in mean values, productivity rate is high
## in clust1 as 1.40, next in clust0 as 1.35 and then in clust2 as 0.55
data.columns
data.dtypes



######## will check in mutlinomial regression on dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import accuracy_score

train, test = train_test_split(data, test_size = 0.30)
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 2:], train.iloc[:, 1])

test_predict = model.predict(test.iloc[:, 2:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,1], test_predict)
### Test accuracy value:- 0.74

train_predict = model.predict(train.iloc[:, 2:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,1], train_predict)
### Train accuracy value:- 0.80


####### will check in  Decision Tree  #######

data.dtypes

data.drop(['clust'], axis = 1, inplace = True)

colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]

#spliting data into training and testing data

from sklearn. model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.30)

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model. fit(train[predictors], train[target])

## Prediction on Test Data
preds = model. predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])
np.mean(preds == test[target]) # Test Data Accuracy 
## test value 0.869

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy
## train value 1.0

# Automatic Tuning - Hyperparameters
######
# GridSearchCV
from sklearn. model_selection import GridSearchCV

model = DT(criterion = 'entropy')
param_grid = {'min_weight_fraction_leaf': [0.1,0.2,0.3,0.4],
              'max_depth': [2,4,6,8,10],
             'max_features': ['log2']}

grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                           scoring = 'accuracy', n_jobs = -1, cv = 5,
                           refit = True, return_train_score = True)

grid_search.fit(train[predictors], train[target])


grid_search.best_params_

cv_dt_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(test[target], cv_dt_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_dt_clf_grid.predict(test[predictors]))
##test value = 0.630
# Evaluation on Training Data
confusion_matrix(train[target], cv_dt_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_dt_clf_grid.predict(train[predictors]))
## Train value = 0.640

# after applied grid search
#test_accuracy = 63.0%
#train_accuracy = 64.0%


# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
model = DT(criterion = 'entropy')
param_dist = {'min_samples_leaf': list(range(1, 1000)),
              'max_depth': list(range(2, 500)),
              'max_features': ['log2']}

n_iter = 50

model_random_search = RandomizedSearchCV(estimator = model,
                                         param_distributions = param_dist,
                                         n_iter = n_iter)

model_random_search. fit(train[predictors], train[target])
model_random_search. best_params_

dT_random = model_random_search.best_estimator_

#prediciton on test data 
pred_random = dT_random.predict(test[predictors])
pd.crosstab(test[target], pred_random, rownames=['Actual'], colnames=['Predictions'])

np.mean(pred_random == test[target])

#predicition on train data 
pred_random = dT_random.predict(train[predictors])
pd.crosstab(train[target], pred_random, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(pred_random == train[target])


# here test accuracy = 0.798
#train accuracy = 0.836


### Will Apply Navie Bayes 
data.dtypes

df = norm_func(data. iloc[:,:])

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
df.dtypes
train_X = data.iloc[:, 1:]
train_y = data.iloc[:, 0]
test_X  = data.iloc[:, 1:]
test_y  = data.iloc[:, 0] 
train , test = train_test_split(data, test_size=0.30, random_state=42)

### Building Multinomial Naive Bays Model
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_X, train_y)

score_multinomial_train = classifier_mb.score(train_X,train_y)
print( score_multinomial_train)
##train value =0.56

score_multinomial_test = classifier_mb.score(test_X,test_y)
print( score_multinomial_test)
## test value = 0.56
### Building Gaussian Naive Bayes Model
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB as GB

classifier_gb = GB()
classifier_gb.fit(train_X, train_y)
score_gaussian_train = classifier_gb.score(train_X,train_y)
print( score_gaussian_train)
## train value = 0.70

score_gaussian_test = classifier_gb.score(test_X,test_y)
print( score_gaussian_test)
## test value = 0.70

data.dtypes
## Multi Linear Regression model 
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('Productivity~Designation+Type_of_workmen+Pulse_rate+Body_Mass_Index+Body_temperature+Work_area+Scope_of_work+Total_hours_worked+Real_Time_Information+Actual_work+Targeted_work+Pending_work+Daily_wage+Site', data = data).fit() # regression model

# Summary
ml1.summary()
# p-values of  features are more than 0.05 so we drop that columns
data.drop(['Designation'], axis = 1, inplace = True)
data.drop(['Type_of_workmen'], axis = 1, inplace = True)
data.drop(['Pulse_rate'], axis = 1, inplace = True)
data.drop(['Work_area'], axis = 1, inplace = True)
data.drop(['Actual_work'], axis = 1, inplace = True)
data.drop(['Pending_work'], axis = 1, inplace = True)
 
ml2 = smf.ols('Productivity~Body_Mass_Index+Body_temperature+Scope_of_work+Total_hours_worked+Real_Time_Information+Targeted_work+Daily_wage+Site', data = data ).fit() # regression model
# Summary
ml2.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml2)

# Variance Inflation Factor (VIF)
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
data.columns
rsq_bmi = smf.ols('Body_Mass_Index ~ Body_temperature + Scope_of_work + Total_hours_worked + Real_Time_Information + Targeted_work + Daily_wage + Site', data = data).fit().rsquared  
vif_bmi = 1/(1 - rsq_bmi)

rsq_temp = smf.ols('Body_temperature ~ Body_Mass_Index + Scope_of_work + Total_hours_worked + Real_Time_Information + Targeted_work + Daily_wage + Site', data = data).fit().rsquared  
vif_temp = 1/(1 - rsq_temp)

rsq_scope = smf.ols('Scope_of_work ~ Body_Mass_Index + Body_temperature + Total_hours_worked + Real_Time_Information + Targeted_work + Daily_wage + Site', data = data).fit().rsquared  
vif_scope = 1/(1 - rsq_scope)

rsq_ttlhrs = smf.ols('Total_hours_worked ~ Body_Mass_Index  + Scope_of_work + Body_temperature + Real_Time_Information + Targeted_work + Daily_wage + Site', data = data).fit().rsquared  
vif_ttlhrs = 1/(1 - rsq_ttlhrs)

rsq_real = smf.ols('Real_Time_Information ~ Body_Mass_Index  + Scope_of_work + Total_hours_worked + Body_temperature + Targeted_work + Daily_wage + Site', data = data).fit().rsquared  
vif_real = 1/(1 - rsq_real)

rsq_trgtwrk = smf.ols('Targeted_work ~ Body_Mass_Index + Scope_of_work + Total_hours_worked + Real_Time_Information + Body_temperature + Daily_wage + Site', data = data).fit().rsquared  
vif_trgtwrk = 1/(1 - rsq_trgtwrk)

rsq_wage = smf.ols('Daily_wage ~ Body_Mass_Index  + Scope_of_work + Total_hours_worked + Real_Time_Information + Targeted_work + Body_temperature + Site', data = data).fit().rsquared  
vif_wage = 1/(1 - rsq_wage)

rsq_site = smf.ols('Site ~ Body_Mass_Index  + Scope_of_work + Total_hours_worked + Real_Time_Information + Targeted_work + Daily_wage + Body_temperature', data = data).fit().rsquared  
vif_site = 1/(1 - rsq_site)

# Storing vif values in a data frame
d1 = {'Variables':['Body_temperature', 'Body_Mass_Index', 'Scope_of_work', 'Total_hours_worked', 'Real_Time_Information', 'Targeted_work', 'Daily_wage', 'type' ], 'VIF':[vif_bmi, vif_temp, vif_scope, vif_ttlhrs, vif_trgtwrk, vif_real, vif_wage, vif_site]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
## VIF values of all  columsn are less than 10,so will continue 
data.columns

# Prediction
pred = ml2.predict(data)

# Q-Q plot
res = ml2.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = data.Productivity, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml2)


### Splitting the data into train and test data 
data_train, data_test = train_test_split(data, test_size = 0.30) # 30% test data
data.columns
# preparing the model on train data 
model_train = smf.ols("Productivity ~ Body_Mass_Index + Body_temperature + Scope_of_work + Total_hours_worked + Real_Time_Information + Targeted_work + Daily_wage + Site", data = data_train).fit()
model_train.summary()
## observed that modle is weak = R- Squared value = 0.461

# prediction on test data set 
test_pred = model_train.predict(data_test)

# test residual values 
test_resid = test_pred - data_test.Productivity
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
## Value = 0.562

# train_data prediction
train_pred = model_train.predict(data_train)

# train residual values
train_resid  = train_pred - data_train.Productivity
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
## 0.532

data.dtypes


### Ensemble techniques using boosting methods ######
# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
df.dtypes
X_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0]
X_test  = data.iloc[:, 1:]
y_test  = data.iloc[:, 0] 

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.30, random_state=0)

#AdaBoostClassifier
# Refer to the links
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(X_test))
accuracy_score(y_test, ada_clf.predict(X_test))
#0.66
# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(X_train))
##0.666




### XGBoostClassifier
# Refer to the links
# https://xgboost.readthedocs.io/en/latest/
# https://xgboost.readthedocs.io/en/latest/python/python_api.html

# pip install xgboost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)


xgb_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(X_test))
accuracy_score(y_test, xgb_clf.predict(X_test))
##96.5

# Evaluation on Training Data
confusion_matrix(y_train, xgb_clf.predict(X_train))
accuracy_score(y_train, xgb_clf.predict(X_train))
##96.5

xgb.plot_importance(xgb_clf)

# GridsearchCV
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'reg_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, y_train)

grid_search.best_params_
cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(X_test))
##81.5

# Evaluation on Training Data with model with hyperparameter
accuracy_score(y_train, cv_xg_clf.predict(X_train))
#81.5

#### KNN Model #####

X = data.iloc[:, 1:]
Y = data.iloc[:, 0] 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Imbalance check
data.Productivity.value_counts()

ytrain = pd.DataFrame(Y_train)
ytest = pd.DataFrame(Y_test)

ytrain.value_counts() / len(ytrain)
ytest.value_counts() / len(ytest)

xtrain = pd.DataFrame(X_train)
xtest = pd.DataFrame(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
####0.707
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
####0.756
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
for i in range(1, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
    
import matplotlib.pyplot as plt #  visualizations 

# train accuracy plot 
plt.plot(np.arange(1, 50, 2), [i[0] for i in acc],"ro-")  

# test accuracy plot
plt.plot(np.arange(1, 50, 2), [i[1] for i in acc],"bo-") 



#### Support Vector Machine ##### 

from sklearn.svm import SVC

train, test = train_test_split(data, test_size = 0.30)
data.dtypes
train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0] 


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear == test_y)
###0.782
pred_train_linear = model_linear.predict(train_X)

np.mean(pred_train_linear == train_y)
##0.780

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf == test_y)
##0.625

pred_train_rbf = model_rbf.predict(train_X)

np.mean(pred_train_rbf == train_y)
##0.572



import pickle

# open the pickle file in writebyte mode
file = open("model2.pkl",'wb')
#dump information to that file
pickle.dump(xgb_clf, file)
file.close()

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
data.columns





