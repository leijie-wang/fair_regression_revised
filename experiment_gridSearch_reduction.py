import numpy as np 
import matplotlib.pyplot as plt 
import shap
import pandas as pd

from fairlearn.metrics import group_summary
from fairlearn.metrics import selection_rate_group_summary

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.reductions import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def demographic(group_summary_adult):
	error_all = group_summary_adult["overall"]
	maxdp = 0
	for error_group in group_summary_adult["by_group"].values():
		dp = abs(error_group-error_all)
		if dp > maxdp:
			maxdp = dp
	return maxdp
random_state = 4
TEST_PORTION = 0.25
ROUND = 1
WEIGHTS = [i/10 for i in range(1,10,2)]
X, y = shap.datasets.adult()#readfrom("adult.data")
y = y * 1
# X = X[:100]
# y = y[:100]
errorlist = [[] for i in range(len(WEIGHTS))]
dplist = [[] for i in range(len(WEIGHTS))]

def evaluate(weight,X_train,y_train,X_test,y_test,sex_train,sex_test,index):
	estimator = GradientBoostingClassifier()
	constraints = DemographicParity()
	gssolver = GridSearch(estimator,constraints,grid_size=10,constraint_weight=weight)
	gssolver.fit(X_train,y_train,sensitive_features=sex_train)
	y_pred = gssolver.predict(X_test)
	# print("y_pred",y_pred)
	group_summary_adult = group_summary(accuracy_score, y_test, y_pred, sensitive_features=sex_test)
	selection_rate_summary = selection_rate_group_summary(y_test, y_pred, sensitive_features=sex_test)
	error = 1 - group_summary_adult["overall"]
	dp = demographic(selection_rate_summary)
	errorlist[index].append(error)
	dplist[index].append(dp)
	print("error:%f,dp:%f"%(error,dp))
for i in range(ROUND):
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_PORTION,random_state=random_state)
	sex_train = X_train['Sex'].apply(lambda sex: "female" if sex == 0 else "male")
	sex_test  = X_test['Sex'].apply(lambda sex: "female" if sex == 0 else "male")
	for index in range(len(WEIGHTS)):
		weight = WEIGHTS[index]
		print("weight:%f, round:%d"%(weight,i))
		evaluate(weight,X_train,y_train,X_test,y_test,sex_train,sex_test,index)
dplist = [ sum(dp)/len(dp) for dp in dplist]
errorlist = [sum(error)/len(error) for error in errorlist]
print("dplist:",dplist)
print("errorlist:",errorlist)
plt.plot(dplist,errorlist,"bo")
plt.show()
