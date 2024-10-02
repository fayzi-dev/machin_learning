import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Loan_data.csv')
# print(df.info())
# print(df.head())

#Encoding
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
# print(df['Gender'].values)
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Dependents'].replace('3+',3, inplace=True)
# print(df['Dependents'])
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
df['Property_Area'] = df['Property_Area'].map({'Semiurban':1, 'Urban':2, 'Rural':3})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# print(df.sample(10))

# Missing Value
# print(df.isnull().sum())
# print(df['Gender'].value_counts())
null_data = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'LoanAmount','Loan_Amount_Term']
df[null_data] = df[null_data].replace({np.nan:df['Gender'].mode(),
                                       np.nan:df['Married'].mode(),                                      
                                       np.nan:df['Dependents'].mode(),
                                       np.nan:df['Self_Employed'].mode(),
                                       np.nan:df['Credit_History'].mode(),
                                       np.nan:df['LoanAmount'].mean(),
                                       np.nan:df['Loan_Amount_Term'].mean()                                   
                                      })
# print(df.isnull().sum())
# print(df.sample(10))

# Spliting Data X,y
X = df.drop(columns=['Loan_ID', 'Loan_Status']).values
y = df['Loan_Status'].values
# print(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)

# Random Forest Model 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion= 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
# print('F1_score_train_Data:', f1_score(y_train, y_pred_train))
# print('F1_score_test_Data:', f1_score(y_test, y_pred_test))

# Built-in feature importance (Gini Importance)

importances = classifier.feature_importances_
feature_imp_df = pd.DataFrame({'importances':importances},index=df.drop(columns=['Loan_ID','Loan_Status']).columns)
feature_imp_df.sort_values(by='importances',ascending=True, inplace=True)           
# print(feature_imp_df)

index = np.arange(len(feature_imp_df))
fig, ax = plt.subplots(figsize=(18,8))
Random_F_C = ax.barh(index, feature_imp_df['importances'], 0.4, color='blue', label= 'Random Forest')
ax.set(yticks=index+0.4, yticklabels=feature_imp_df.index)
ax.legend()
# plt.show()


#Random Forest Regression
data = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Position_Salaries.csv')
# print(data.head())
# print(data.info())

X = data.iloc[:,1:2].values
y = data.iloc[:, 2].values

# print(X, y)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 100, random_state= 0)
regressor.fit(X,y)

y_pred = regressor.predict([[5.6]])
print(y_pred)

#Graph
x_grid =np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(X, y ,color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()