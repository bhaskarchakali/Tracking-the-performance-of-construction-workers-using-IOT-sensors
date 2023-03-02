
import pyodbc #python open database connectivity
import pandas as pd  # for data manipulation
import pandas_profiling # to perform automated EDA

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-A0U077F\SQLEXPRESS;'
                      'Database=construction data;'
                      'Trusted_Connection=yes;')

labour = pd.read_sql_query('SELECT * FROM [dbo].[Sheet1$]', conn)

pd.set_option('display.max_columns',None)

print(labour)
print(type(labour))

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

labour.shape

labour.head()


labour.isna().sum()
# No null values Present


labour.duplicated()

# No Duplicate Values Present

#To view the categorical and numerical columns and its datatypes in the dataset                                    
labour.info()           # 10 numerical and 12 categorical features

# Encoding all the ordinal columns and creating a dummy variable for them to see if there are any effects on Performance Rating
from sklearn.preprocessing import LabelEncoder, StandardScaler
enc = LabelEncoder()
for i in (2,3,4,5,6,11,12,13,14,15,20,21):
    labour.iloc[:,i] = enc.fit_transform(labour.iloc[:,i])
labour

# Data Cleaning
#Dropping the columns not necessary for analysis
labour=labour.drop(['Name','Work Date','Rfid','Latitude','Longitude','Temperature','BVP'],axis=1)
labour


labour.info()

#Converting datatype into another format.
labour.Galvanic_Skin_Response_Sensor = labour.Galvanic_Skin_Response_Sensor.astype('float32')
labour.Attendance = labour.Attendance.astype('int64')
labour.RR = labour.RR.astype('float32')
labour.HR = labour.HR.astype('float32')
labour.info()


#boxplot - checking presence of outliers
fig=plt.figure(figsize=(30,5))
sns.boxplot(data= labour)
plt.show()

# Age and Experience has Outliers

sns.histplot(data=labour, x='Work_Position', kde=True)

sns.histplot(data=labour, x='Experience', kde=True)


# histogram
labour.hist() # overall distribution of data


# calculating TWSS - Total within SS using different cluster range
from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(labour)
    TWSS.append(kmeans.inertia_)
    
TWSS


# Plotting the Scree plot using the TWSS from above defined function
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# Selecting 4 clusters from the above scree plot which is the optimum number of clusters, 
# as the curve is seemingly bent or showinf an elbow format at K = 4

model = KMeans(n_clusters = 3)
model.fit(labour)

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
labour['Performance'] = mb


labour.iloc[:, 1:].groupby(labour.Performance).mean()

labour

#To check imbalance 
labour['Performance'].value_counts() 
#3 good, average, poor - 3 classes in taget feature

#Correlation 
corrMatrix = labour.corr()
corrMatrix



plt.pie(labour.Work_Position.value_counts(), autopct='%1.f%%', pctdistance=1.5)
plt.title('Work_Position')  # imbalanced dataset


# Correlation between different variables
corr = labour.corr()
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
#Attendance and working hrs per day have strong correlation
#netconnectivity and workfromhome has strong correlation
#performance rating has positive and negative correlation with the features


#Automatic EDA : Pandas profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(labour)
profile


X = labour.drop(["Performance"],axis=1)
X



Y = labour["Performance"]
Y

# splitting the data into testing and training data.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

# Training the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]

model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)
model_gridrf.fit(X_train,Y_train)


model_gridrf.best_params_



# Predicting the model
Y_predict_rf = model_gridrf.predict(X_test)



# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(Y_test,Y_predict_rf))
print(classification_report(Y_test,Y_predict_rf))



confusion_matrix(Y_test,Y_predict_rf)


import pickle

# open the pickle file in writebyte mode
file = open("constructionmodel.pkl",'wb')
#dump information to that file
pickle.dump(model_gridrf, file)
file.close()

# Loading model to compare the results
model = pickle.load(open('constructionmodel.pkl','rb'))





