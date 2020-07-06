"""
Augment the dataset according to the loss functions.

Input:
- a regression data set (x, a, y), which may be obtained using the data_parser
- loss function
- Theta, a set of thresholds in between 0 and 1

Output:
a weighted classification dataset (X, A, Y, W)
"""

import functools
import numpy as np
import pandas as pd
import fairlearn.regression.data_parser as parser
from itertools import repeat
import itertools
_LOGISTIC_C = 5

def augment_predX(X,Theta):
    """
    Takes input data and augment it with an additional feature of
    theta; Return: X tensor_product Theta
    For absolute loss, we don't do any reweighting.  
    TODO: might add the alpha/2 to match with the write-up
    """
    n = np.shape(X)[0]
    num_theta = len(Theta)
    X_aug = pd.concat(repeat(X, num_theta))
    #create muplicates of X, A and Y.
    #in an order that x1,x2,x3,...,xn,x1,x2,...
    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)
    X_aug.index = range(n * num_theta)

    return X_aug

def augment_data_ab(X, A, Y, Theta):
    """
    Takes input data and augment it with an additional feature of
    theta; Return: X tensor_product Theta
    For absolute loss, we don't do any reweighting.  
    TODO: might add the alpha/2 to match with the write-up
    """
    n = np.shape(X)[0]
    num_theta = len(Theta)
    X_aug = pd.concat(repeat(X, num_theta))
    A_aug = pd.concat(repeat(A, num_theta))
    Y_values = pd.concat(repeat(Y, num_theta))
    #create muplicates of X, A and Y.
    #in an order that x1,x2,x3,...,xn,x1,x2,...
    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

    Y_aug = Y_values >= X_aug['theta']
    #compare the score with its given threshold
    Y_aug = Y_aug.map({True: 1, False: 0})
    X_aug.index = range(n * num_theta)
    Y_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    #why changing the index
    W_aug = pd.Series(1, Y_aug.index)
    #identical weights for each instance.
    return X_aug, A_aug, Y_aug, W_aug


def augment_data_sq(x, a, y, Theta):
    """
    Augment the dataset so that the x carries an additional feature of theta
    Then also attach appropriate weights to each data point.

    Theta: Assume uniform grid Theta
    """
    n = np.shape(x)[0]  # number of original data points
    num_theta = len(Theta)#the number of thetas, that is, N or 1/alpha in the paper
    width = Theta[1] - Theta[0]#alpha
    X_aug = pd.concat(repeat(x, num_theta))
    A_aug = pd.concat(repeat(a, num_theta))
    Y_values = pd.concat(repeat(y, num_theta))

    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)
    ##until here, exactly the same to the augment_data_ab function

    X_aug.index = range(n * num_theta)
    # Y_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    Y_values.index = range(n * num_theta)

    # two helper functions
    sq_loss = lambda a, b: (a - b)**2  # square loss function
    weight_assign = lambda theta, y: (sq_loss(theta + width/2, y) - sq_loss(theta - width/2, y))
    #this weight corresponds to the c(y,z) = N(l(y,z+alpha/2)-l(y,z-alpha)). 
    #however, how does this releta to the paper? 
    #By first formulate the data (X,A,Y) to the form ((X,Z),A,C), where C corresponds to the c(y,z)
    W = weight_assign(X_aug['theta'], Y_values)
    Y_aug = 1*(W < 0)
    W = abs(W)
    # Compute the weights
    return X_aug, A_aug, Y_aug, W


def augment_data_logistic(x, a, y, Theta):
    """
    Augment the dataset so that the x carries an additional feature of theta
    Then also attach appropriate weights to each data point, so that optimize
    for logisitc loss
    
    Theta: Assume uniform grid Theta
    y: assume the labels are {0, 1}
    """
    n = np.shape(x)[0]  # number of original data points
    num_theta = len(Theta)
    width = Theta[1] - Theta[0]
    X_aug = pd.concat(repeat(x, num_theta))
    A_aug = pd.concat(repeat(a, num_theta))
    Y_values = pd.concat(repeat(y, num_theta))

    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

    X_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    Y_values.index = range(n * num_theta)
    ##until here, exactly the same to the augment_data_ab function
    
    # two helper functions
    logistic_loss = lambda y_hat, y: np.log(1 + np.exp(-(_LOGISTIC_C)*(2 * y - 1) * (2 * y_hat - 1))) / (np.log(1 + np.exp(_LOGISTIC_C)))  
    # re-scaled logistic loss
    ##in accordance with the cost function in the paper
    #logistic_loss = lambda y_hat, y: np.log(1 + np.exp(-(_LOGISTIC_C)*(2 * y - 1) * (2 * y_hat - 1)))  # re-scaled logistic loss
    weight_assign = lambda theta, y: (logistic_loss(theta + width/2,
                                                    y) - logistic_loss(theta - width/2, y))
    W = weight_assign(X_aug['theta'], Y_values)
    Y_aug = 1*(W < 0)
    #a negative cost means that l(y,z+alpha/2) < l(y,z-alpha/2), which means that the true class is z+alpha/2, so it is 1.
    W = abs(W)
    # Compute the weights
    return X_aug, A_aug, Y_aug, W
