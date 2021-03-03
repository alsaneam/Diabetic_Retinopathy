# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 00:03:46 2020

@author: Dr. M Alauthman
"""

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X,y)
print(clf.feature_importances_)

from xgboost import XGBClassifier
from matplotlib import pyplot
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()