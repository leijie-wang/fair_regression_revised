"""
Run the exponentiated gradient method for training a fair regression
model.

Input:
- (x, a, y): training set
- eps: target training tolerance
- Theta: the set of Threshold
- learner: the regression/classification oracle 
- constraint: for now only handles demographic parity (statistical parity)
- loss: the loss function

Output:
- a predictive model (a distribution over hypotheses)
- auxiliary model info

"""
from fairlearn._input_validation import _validate_and_reformat_input
import fairlearn.regression.data_parser as parser
import fairlearn.regression.data_augment as augment
import fairlearn.regression.eval as evaluate
from fairlearn.reductions._moments.conditional_selection_rate import DemographicParity_Theta
from fairlearn.reductions._exponentiated_gradient.exponentiated_gradient import ExponentiatedGradient
import pandas as pd

from sklearn.base import BaseEstimator, MetaEstimatorMixin
def _mean_pred(dataX, hs, weights):
    # Return a weighted average of predictions produced by classifiers in hs
    
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](dataX)
    return pred[weights.index].dot(weights)

class FairRegression(BaseEstimator, MetaEstimatorMixin):

	def __init__(self,eps,Theta,estimator,constraints="DP", loss="square"):
		self.estimator = estimator
		self.constraints = constraints
		self.eps = eps
		self.loss = loss
		self.Theta = Theta

	def fit(self,x,y,**kwargs):
		_, y_train, sensitive_features = _validate_and_reformat_input(
            x, y, enforce_binary_labels=False, **kwargs)
		if self.loss == "square": # squared loss reweighting
			X, A, Y, W = augment.augment_data_sq(x, sensitive_features, y_train, self.Theta)
		elif self.loss == "absolute":  # absolute loss reweighting (uniform)
			X, A, Y, W = augment.augment_data_ab(x, sensitive_features, y_train, self.Theta)
		elif self.loss == "logistic":  # logisitic reweighting
			X, A, Y, W = augment.augment_data_logistic(x, sensitive_features, y_train, self.Theta)
		else:
			raise Exception('Loss not supported: ', str(loss))
		if self.constraints == "DP":  # DP constraint
			self.constraints = DemographicParity_Theta()
			self.expgrad = ExponentiatedGradient(self.estimator,self.constraints,self.eps,error_weights=W)
			self.expgrad.fit(X,Y,sensitive_features=A)
			self.weights_ = self.expgrad.weights_
			self.best_classifier = lambda X : _mean_pred(X, self.expgrad._hs, self.expgrad.weights_)
			self._hs = self.expgrad._hs
			self.predictors_ = self.expgrad.predictors_
			self.best_gap_ = self.expgrad.best_gap_
			self.last_iter_ = self.expgrad.last_iter_
			self.best_iter_ = self.expgrad.best_iter_
			self.n_oracle_calls_= self.expgrad.n_oracle_calls_
			self.n_classifiers = len(self._hs)
		else:  # exception
			raise Exception('Constraint not supported: ', str(constraint))
	def predict(self,x):
		X = augment.augment_predX(x,self.Theta)
		#first make sure that these two are of the same length
		off_set = len(self._hs) - len(self.weights_)
		if (off_set > 0):
			off_set_list = pd.Series(np.zeros(off_set), index=[i +len(self.weights_) for i in range(off_set)])
			result_weights = self.weights_.append(off_set_list)
		else:
			result_weights = self.weights_

		hs = self._hs[result_weights > 0]
		result_weights = result_weights[result_weights > 0]
		num_t = len(self.Theta)
		num_h = len(hs)
		n = int(len(X) / num_t)
	    #the number of original examples.

	    # predictions
		pred_list = [pd.Series(evaluate.extract_pred(X, h(X), self.Theta),
	                           index=range(n)) for h in hs]
		total_pred = pd.concat(pred_list, axis=1, keys=range(num_h)) #lists of predictions for different hs.
		#prediction = pd.DataFrame(np.dot(total_pred,pd.DataFrame(result_weights)))
		#print(prediction)
		#return prediction
		return total_pred, result_weights
