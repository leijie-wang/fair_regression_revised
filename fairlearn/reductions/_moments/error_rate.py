# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

#commented

import pandas as pd
from .moment import ClassificationMoment
from .moment import _ALL, _LABEL
_WEIGHT = "weights"

class ErrorRate(ClassificationMoment):
    """Misclassification error."""

    short_name = "Err"

    def load_data(self, X, y, weights = None, **kwargs):
        #print("I am in revised fairlearn errorrate")
        """Load the specified data into the object."""
        super().load_data(X, y, **kwargs)
        self.index = [_ALL] #“all”
        if weights is None: #the value is "sensitive_features"
            self.tags[_WEIGHT] = 1
        else:
            self.tags[_WEIGHT] = weights

    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags[_WEIGHT]*(self.tags[_LABEL] - pred).abs()).mean(),
                          index=self.index)#here self.tags = y is the true label.
        #run as pd.Series(data=abs(self.tags[_LABEL] - pred).mean(),
        #                  index=self.index)
        self._gamma_descr = str(error)
        #what is this for? the same appears in the regression algorithm. 
        return error

    # def project_lambda(self, lambda_vec):
    #     """Return the lambda values."""
    #     return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        if lambda_vec is None:
            return self.tags[_WEIGHT] * (2 * self.tags[_LABEL] - 1)
        #why? I thought it would be 0 or 1.
        else:
            return lambda_vec[_ALL] * self.tags[_WEIGHT] * (2 * self.tags[_LABEL] - 1)
        #why do we need to multiply this factor?

