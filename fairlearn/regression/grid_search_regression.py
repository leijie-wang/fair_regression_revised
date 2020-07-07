"""
Run the grid search method for training a fair regression
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
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from fairlearn._input_validation import _validate_and_reformat_input
import fairlearn.regression.data_parser as parser
import fairlearn.regression.data_augment as augment
import fairlearn.regression.eval as evaluate
from fairlearn.reductions._moments.conditional_selection_rate import DemographicParity_Theta

from fairlearn.reductions._moments import Moment, ClassificationMoment
from fairlearn.reductions._grid_search.grid_search import TRADEOFF_OPTIMIZATION, GridSearch


class GridSearchRegression(BaseEstimator, MetaEstimatorMixin):

	def __init__(self,estimator,
					Theta,
					constraints="DP",
					loss="square",
					selection_rule=TRADEOFF_OPTIMIZATION,
					constraint_weight=0.5,
					grid_size=10,
					grid_limit=2.0,
					grid_offset=None,
					grid=None):
		#regression problem related parameters
		self.estimator = estimator
		self.constraints = constraints
		self.loss = loss
		self.Theta = Theta
		#grid search related parameters
		self.selection_rule = selection_rule
		self.constraint_weight = constraint_weight
		self.grid_size = grid_size
		self.grid_limit = grid_limit
		self.grid_offset = grid_offset
		self.grid = grid
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
			self.grid_search = GridSearch(self.estimator,
											self.constraints,
											self.selection_rule,
											self.constraint_weight,
											self.grid_size,
											self.grid_limit,
											self.grid_offset,
											self.grid,
											W)
			self.grid_search.fit(X,Y,sensitive_features=A)
		else:  # exception
			raise Exception('Constraint not supported: ', str(constraint))
	def predict(self,x):
		X = augment.augment_predX(x,self.Theta)
		pred_list = [pd.Series(evaluate.extract_pred(X,self.grid_search.predict(X),self.Theta))]
		total_pred = pd.concat(pred_list, axis=1, keys=range(1))
		return total_pred
