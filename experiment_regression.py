import functools
from scipy.stats import norm
import numpy as np
import pandas as pd
import fairlearn.regression.data_parser as parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import fairlearn.regression.eval as evaluate
import fairlearn.regression.solvers as solvers
from fairlearn.regression.exp_grad import FairRegression
import fairlearn.regression.data_augment as augment
print = functools.partial(print, flush=True)


def train_test_split_groups(x, a, y, random_seed):
    """Split the input dataset into train and test sets

    TODO: Need to make sure both train and test sets have enough
    observations from each subgroup
    """
    # size of the training data
    groups = list(a.unique())
    x_train_sets = {}
    x_test_sets = {}
    y_train_sets = {}
    y_test_sets = {}
    a_train_sets = {}
    a_test_sets = {}

    for g in groups:
        x_g = x[a == g]
        a_g = a[a == g]
        y_g = y[a == g]
        x_train_sets[g], x_test_sets[g], a_train_sets[g], a_test_sets[g], y_train_sets[g], y_test_sets[g] = train_test_split(x_g, a_g, y_g, test_size=TEST_SIZE, random_state=random_seed)
    ## it seems that train_test_split function can be used to more than one datasets, and split them in accordance 
    x_train = pd.concat(x_train_sets.values())
    x_test = pd.concat(x_test_sets.values())
    y_train = pd.concat(y_train_sets.values())
    y_test = pd.concat(y_test_sets.values())
    a_train = pd.concat(a_train_sets.values())
    a_test = pd.concat(a_test_sets.values())
    ## combine data of different groups together.

    # resetting the index
    x_train.index = range(len(x_train))
    y_train.index = range(len(y_train))
    a_train.index = range(len(a_train))
    x_test.index = range(len(x_test))
    y_test.index = range(len(y_test))
    a_test.index = range(len(a_test))
    return x_train, a_train, y_train, x_test, a_test, y_test
def evaluate_accuracy_fairness(y_true,a,total_pred,result_weights,loss,Theta):

    pred_group = evaluate.extract_group_pred(total_pred, a)
    weighted_loss_vec = evaluate.loss_vec(total_pred, y_true, result_weights, loss)
    #print("weighted_loss_vec",weighted_loss_vec)
    # Fit a normal distribution to the sq_loss vector
    loss_mean, loss_std = norm.fit(weighted_loss_vec)
    #print("loss_mean:",loss_mean)
    #print("loss_std:",loss_std)
    # DP disp
    PMF_all = evaluate.weighted_pmf(total_pred, result_weights, Theta)
    ## probably understood as the probability of each theta-threshold.
    PMF_group = [evaluate.weighted_pmf(pred_group[g], result_weights, Theta) for g in pred_group]
    ## probably understood as the probability of each theta-threshold inside each protected group
    DP_disp = max([evaluate.pmf2disp(PMF_g, PMF_all) for PMF_g in PMF_group])
    ## calculate the maximum gamma(a,z) as the statistical parity for this classifier.
    if loss == "square":
        return DP_disp,np.sqrt(loss_mean)
    else:
        return DP_disp,loss_mean
dataset = "communities"
_SMALL = True
size = 200
DATA_SPLIT_SEED = 4
# Global Variables
TEST_SIZE = 0.5  # fraction of observations from each protected group
Theta = np.linspace(0, 1.0, 41)
## the construction of thresholds, it would be better to set 40 as a constant parameter.
alpha = (Theta[1] - Theta[0])/2
eps_list = [0.275, 0.31, 1]
#eps_list = [0.275]
constraint = "DP"
loss = "square"

if dataset == 'law_school':
    x, a, y = parser.clean_lawschool_full()
elif dataset == 'communities':
    x, a, y = parser.clean_communities_full()
elif dataset == 'adult':
    x, a, y = parser.clean_adult_full()
else:
    raise Exception('DATA SET NOT FOUND!')
if _SMALL:
    x, a, y = parser.subsample(x, a, y, size)

x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

fair_model = {}
train_results = {}
test_results = {}
print("dataset:",dataset)
print("Small Sample:",_SMALL)
print("loss:",loss)

for eps in eps_list:
    
    estimator = solvers.LeastSquaresLearner(Theta)
    #estimator = solvers.SVM_LP_Learner(off_set=alpha)
    #estimator = solvers.LogisticRegressionLearner(Theta)
    #estimator = solvers.RF_Regression_Learner(Theta)
    fair_model[eps] = FairRegression(eps,Theta,estimator,constraint,loss)
    fair_model[eps].fit(x_train,y_train,sensitive_features=a_train)
    #y_pred_train = fair_model[eps].predict(x_train)
    y_pred_train,result_weights_train = fair_model[eps].predict(x_train)
    dp_train,loss_train = evaluate_accuracy_fairness(y_train,a_train,y_pred_train,result_weights_train,loss,Theta)
    train_results[eps] = [dp_train,loss_train]
    
    #print("y_pred_train:",y_pred_train)
    #print("result_weights_train:",result_weights_train)
    #print("eps:%f: dp_train: %f loss_train:%f"%(eps,dp_train,loss_train))
    
    #y_pred_test = fair_model[eps].predict(x_test)
    y_pred_test, result_weights_test= fair_model[eps].predict(x_test)
    dp_test,loss_test = evaluate_accuracy_fairness(y_test,a_test,y_pred_test,result_weights_test,loss,Theta)
    test_results[eps] = [dp_test,loss_test]

    #print("y_pred_test:",y_pred_test)
    #print("result_weights_test:",result_weights_test)
    #print("eps:%f: dp_test: %f loss_test:%f"%(eps,dp_test,loss_test))
for eps in eps_list:
    print("eps:%f: dp_train: %f loss_train:%f"%(eps,train_results[eps][0],train_results[eps][1]))
for eps in eps_list:
    print("eps:%f: dp_test: %f loss_test:%f"%(eps,test_results[eps][0],test_results[eps][1]))

