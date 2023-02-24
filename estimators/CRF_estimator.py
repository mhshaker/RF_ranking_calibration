import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.calibration import _SigmoidCalibration

class CRF_calib(BaseEstimator, ClassifierMixin):

    def __init__(self, r=0.5):
        # r can be between 0 and 1
        # including 0 and 1
        # not including 1, if we want the AUC not to be affected
        self.r = r

    def fit(self, X, y): # X,y are calibration dataset

        # [Goal] set the value of "r"
        # learn _SigmoidCalibration "a" and "b"
        # optimize _SigmoidCalibration with brier_score_loss (from sklearn.metrics import brier_score_loss)

        self.sig = _SigmoidCalibration().fit(X, y)



        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # if pi = Max({p1,..., pk})
        max_idx = np.argmax(X)
        X[max_idx] = X[max_idx] + self.r * (1-X[max_idx])
        # otherwise
        X[:max_idx] = X[:max_idx] * (1-self.r)
        X[max_idx+1:] = X[max_idx+1:] * (1-self.r)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        r = self.sig.predict(X)

        # max(X) + r
        # rest(X) - r

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]