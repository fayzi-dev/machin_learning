
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

df =  pd.read_csv("Datasets/Churn_Modelling.csv")
df.head()

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values
X

"""**Label Encoding the categorecal data**"""

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X[:, 2] = label.fit_transform(X[:, 2])
X

"""**One Hot Encoding whit ColumnTransformer**"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
col_trans = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(col_trans.fit_transform(X))
X

"""Spliting dataset into the Train and Test

"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 0 )
X_test

from sklearn.preprocessing import StandardScaler
standard_df = StandardScaler()
X_train = standard_df.fit_transform(X_train)
X_test = standard_df.transform(X_test)
X_test

"""Bulding The ANN(Artificial Nerual Network)"""

ANN = tf.keras.models.Sequential() #Initializing the ANN

ANN.add(tf.keras.layers.Dense(units=6, activation='relu')) #Add input layer
ANN.add(tf.keras.layers.Dense(units=6, activation='relu')) #Add hidden layer
ANN.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #Add output layer

ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #Compiling the ANN

ANN.fit(X_train,y_train, batch_size=32, epochs=100)

new_data = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
new_data = standard_df.transform(new_data)
pred = ANN.predict(new_data)
if(pred > 0.5):
  print("True") 
else:
  print("False") 
