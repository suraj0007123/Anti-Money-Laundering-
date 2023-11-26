

import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Importing Dataset

amldata = pd.read_csv(r'D:\AML\Money_Laundering_Dataset.csv')
amldata.shape

amldata.info()

amldata.describe()

amldata.head()

### Measure of Central Tendency 

amldata.mean()

amldata.median()

amldata.mode()

### Measure of Dispersion/Spread

amldata.var()

### Skewnes

amldata.skew()

## Kurtosis

amldata.kurt()

amldata=amldata.drop(["Unnamed: 0"],axis=1)

amldata.isna().sum()

amldata["step"].fillna(amldata["step"].median(), inplace = True)

### nameOrig
amldata["nameOrig"].fillna(amldata["nameOrig"].mode(), inplace = True)

###NewbalanceOrig
value3=amldata['newbalanceOrig']+amldata["amount"]
amldata['oldbalanceOrg'].fillna(value3, inplace = True)

### OldbalanceOrg
value2 = amldata['oldbalanceOrg'] - amldata["amount"]
amldata['newbalanceOrig'].fillna(value2, inplace = True)

##nameDest
amldata['nameDest'].fillna(amldata['nameDest'].mode(),  inplace = True)

##oldbalanceDest
amldata['oldbalanceDest'].fillna(amldata['oldbalanceDest'].mean(),  inplace = True)


##isFraud
amldata['isFraud'].fillna(amldata['isFraud'].mode()[0],inplace=True)
##isFraud
amldata['isFlaggedFraud'].fillna(amldata['isFlaggedFraud'].mode()[0],inplace=True)

amldata.isna().sum()

"""Duplicate values"""

amldata.duplicated().sum()  ## no duplicate values



# High amount
## Here finding the supspicous Transaction which above thrdhold amount 
amldata['high'] = [1 if n>250000 else 0 for n in amldata['amount']]

amldata.head()

# Rapid Movement

## Transaction frequency for benficier account
''' 1 if the receiver receives money from many individuals else it will be 0'''
amldata['rapid']=amldata['nameDest'].map(amldata['nameDest'].value_counts())
amldata['Rapid']=[1 if n>30 else 0 for n in amldata['rapid']]
amldata.drop(['rapid'],axis=1,inplace = True)

''' customer ids which starts with C in Receiver name for cash_outs'''

def label_customer (row):
    if(row['nameDest'] and isinstance(row['nameDest'], str)):
        if row['type'] == 'CASH_OUT' and 'C' in row['nameDest']:
            return 1
    return 0
    
amldata['merchant'] = amldata.apply (lambda row: label_customer(row), axis=1)
amldata['merchant'].fillna(0,inplace=True)
amldata.head()


df = amldata.copy()

"""Balancing DataSet"""

df.head()

# One hot encoding
df =pd.concat([df,  pd.get_dummies(df['type'],    prefix='type_'  )],axis=1)
df.drop(['type'],  axis=1,  inplace = True)

df.head()

df.drop(['nameOrig', 'nameDest'], axis = 1, inplace = True)

#Normalization of  the numerical columns
col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest']

def norm(i):
  x=(i-i.min())/(i.max()-i.min())
  return x

df[col_names]=norm(df[col_names])

df.head()

"""# **Model Building**

**Splitting Data**
"""

X = df.drop('isFraud', axis=1).values
Y = df['isFraud'].values

X.shape

Y.shape



X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, random_state = 111)


# Handling class imbalance using SMOTE based techniques


from imblearn.over_sampling import SMOTE

# oversampling the train dataset using SMOTE
smote = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
SX, Sy= smote.fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split(SX, Sy, train_size = 0.8, random_state = 111)



## DecisionTreeClassifier
dtc =DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
dtc_score=accuracy_score(y_test,y_pred)
dtc_conf=confusion_matrix(y_test,y_pred)
test_acc=accuracy_score(y_test,y_pred)
train_acc=accuracy_score(y_train,dtc.predict(X_train))
print("Test acccuracy: ",test_acc*100)
print("Train acccuracy: ",train_acc*100)
f1=f1_score(y_test,y_pred)
print("f1 score: ",f1)

### Pickle file

###### Creating Pickle File  ##########
import pickle
pickle.dump(dtc,open("project.pkl","wb"))



loaded_model=pickle.load(open("project.pkl","rb"))
output=loaded_model.predict(X_test)
