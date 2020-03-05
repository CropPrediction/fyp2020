# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:13:23 2020

@author: lavan
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


headers=["Village","Texture","Slope_Code","Erosion","Gravel","Rocky_Code","LCC","Drainage","Depth"]
df = pd.read_csv("find_alg.csv",header=None, names=headers, na_values="?" )
#print(df.head())
#print(df.dtypes)
#obj_df = df.select_dtypes(include=['object']).copy()
#print(obj_df.dtypes)
#array = df.values
#X = array[:,1:]
#Y = array[:,0]
X=df.iloc[:,1:]
Y=df.iloc[:,0:1]
# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()
#print(Y)

# apply le on categorical feature columns
X['Texture'] = le.fit_transform(X['Texture'])
X['Slope_Code'] = le.fit_transform(X['Slope_Code'])
X['Gravel'] = le.fit_transform(X['Gravel'])
X['Rocky_Code'] = le.fit_transform(X['Rocky_Code'])
X['LCC'] = le.fit_transform(X['LCC'])
X['Drainage'] = le.fit_transform(X['Drainage'])
X['Depth'] = le.fit_transform(X['Depth'])
Y['Village']=le.fit_transform(Y['Village'])
#X['Texture'] = X['Texture'].apply(lambda col: le.fit_transform(col))
df['Texture']=le.fit_transform(df['Texture'])
df['Slope_Code'] = le.fit_transform(df['Slope_Code'])
df['Gravel'] = le.fit_transform(df['Gravel'])
df['Rocky_Code'] = le.fit_transform(df['Rocky_Code'])
df['LCC'] = le.fit_transform(df['LCC'])
df['Drainage'] = le.fit_transform(df['Drainage'])
df['Depth'] = le.fit_transform(df['Depth'])
df['Village']=le.fit_transform(df['Village'])


df.plot(x="Depth", y="Village", style='o')  
plt.title('Texture vs village')  
plt.xlabel('Texture')  
plt.ylabel('Village')  
plt.show()


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()