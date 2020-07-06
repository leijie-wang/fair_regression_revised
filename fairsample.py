import numpy as np 
import matplotlib.pyplot as plt 
import shap
import pandas as pd
import fairlearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from fairlearn.widget import FairlearnDashboard
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
'''def readfrom(filename):
	data = dict()
	y_true = list()
	items = ['Age', 'Workclass', 'Education-Num','Marital Status', 'Occupation', 
	'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']
	properties = dict()
	for item in items:
		data[item] = list()
		properties[item] = dict()
	file = open(filename)
	for line in file.readlines():
		line = line.split(',')
		line = line[:3]+line[5:]
		#print(line)
		if len(line) < len(items):
			continue
		for index in range(len(items)):
			char = line[index]
			char = char.strip()
			item = items[index]
			if char.isdigit():
				char = int(char)
			else:
				if char in properties[item]:
					char = properties[item][char]
				else:
					count = len(properties[item])
					properties[item][char] = count
					char = count
			data[item].append(char)
		if line[-1].strip() == ">50K":
			y_true.append(1)
		else:
			y_true.append(0)
	# print(data["Sex"][:10])
	# print(y_true[:100])
	return pd.DataFrame(data), np.array(y_true)'''


X, y_true = shap.datasets.adult()#readfrom("adult.data")
y_true = y_true * 1
sex = X['Sex'].apply(lambda sex: "female" if sex == 0 else "male")

classifier = DecisionTreeClassifier()
classifier.fit(X,y_true)

y_pred = classifier.predict(X)
result1 = metrics.group_summary(accuracy_score, y_true,y_pred,sensitive_features=sex)
print("group_summary",result1)
result2 = metrics.selection_rate_group_summary(y_true, y_pred, sensitive_features=sex)
print("selection_rate_group_summary",result2)
# FairlearnDashboard(sensitive_features=sex,
#                        sensitive_feature_names=['sex'],
#                        y_true=y_true,
#                        y_pred={"initial model": y_pred})

np.random.seed(0)
constraint = DemographicParity()
classifier = DecisionTreeClassifier()
mitigator = ExponentiatedGradient(classifier,constraint)
#print("constructing mitigator")
mitigator.fit(X,y_true,sensitive_features=sex)
y_pred_mitigated = mitigator.predict(X)
result2_mitigated = metrics.selection_rate_group_summary(y_true, y_pred_mitigated, sensitive_features=sex)
print("selection_rate_group_summary mitigated",result2_mitigated)
FairlearnDashboard(sensitive_features=sex, sensitive_feature_names=['sex'],y_true=y_true,y_pred={"initial model": y_pred, "mitigated model": y_pred_mitigated})
#FairlearnDashboard(sensitive_features=sex, sensitive_feature_names=['sex'],y_true=y_true,y_pred={"initial model": y_pred})


