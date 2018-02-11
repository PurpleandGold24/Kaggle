# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:16:51 2018

@author: chenxi11
"""

from __future__ import print_function
import csv
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics

###data loading
print(">start data loading..")
def load_tsv(path, hasColname, sep = '\t'):
        
#    with open(path, encoding='utf-8') as tsvfile:
    with open(path, 'rb') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=sep)
        mydataList = []
        for row in reader:
            mydataList.append(row)            
    
    if hasColname:
        return pd.DataFrame(mydataList[1:], columns = mydataList[0])
    else:
        return pd.DataFrame(mydataList)

mypath = '/data/chenxi11/mydata/Kaggle/MercariPriceSuggestionChallenge/'
#a = pd.DataFrame.from_csv(mypath + 'train.tsv', sep='\t')
train = load_tsv(mypath + 'train.tsv', True)
test = load_tsv(mypath + 'test.tsv', True)


###data preprocessing
print(">start data preprocessing..")
#change data type
train.price = train.price.astype(float)

#splitting categories
sc = train.copy(deep=True)
sc['speCount'] = sc['category_name'].apply(lambda x: str.count(x, '/'))

def subcategories(df, num, colname1, colname2, sep):
    
    subcDict = {'Cate_'+str(i):[] for i in range(num)}
    
    nrows = df.shape[0]
    for i in range(nrows):
        if df.loc[i, colname2] == num:
            slist = train.loc[i, colname1].split(sep)
            for j in range(len(slist)):
                subcDict['Cate_'+str(j)].append(slist[j])
                
        else:
            slist = train.loc[i, colname1].split(sep)
            for j in range(num):
                if j < len(slist):
                    subcDict['Cate_'+str(j)].append(slist[j])            
                else:
                    subcDict['Cate_'+str(j)].append('')
                    
    return subcDict 

temp = subcategories(sc, 5, 'category_name', 'speCount', '/')
sc['Cate_0'] = temp['Cate_0']
sc['Cate_1'] = temp['Cate_1']
sc['Cate_2'] = temp['Cate_2']
sc['Cate_3'] = temp['Cate_3']
sc['Cate_4'] = temp['Cate_4']
selectCols = ['item_condition_id', 'shipping', 'Cate_0', 'Cate_1', 'price']
dat = sc[selectCols]

#convert to dummies
cateCols = ['item_condition_id', 'shipping','Cate_0', 'Cate_1']
dat = pd.get_dummies(dat, cateCols)
print(dat.shape)

#model training
print(">start model training..")
#clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#           oob_score=False, random_state=0, verbose=0, warm_start=False)
#gridsmodel = GridSearchCV(estimator=clf, param_grid={'n_estimators':[50,70], 'max_depth':range(3,11,2)},
#                                                     cv=5, scoring='accuracy',verbose=1, n_jobs=-1)
#gridsmodel.fit(train_x, train_y)
#print('best score is:', str(gridsmodel.best_score_))
#print('best params are:', str(gridsmodel.best_params_))
clf = RandomForestRegressor(max_depth=2, random_state=0)


train_x, test_x, train_y, test_y = train_test_split(np.array(dat.drop(['price'], axis=1)),
                                                    np.array(dat['price']), test_size = 0.2, random_state = 2018)

clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
predict_y = clf.predict(test_x)
mse = metrics.mean_squared_error(test_y, predict_y)
print("mse is %f"%mse)
