import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.calibration import _SigmoidCalibration

class CRF_calib(BaseEstimator, ClassifierMixin):

    def fit(self, X, y): # X,y are calibration dataset

        # [Goal] set the value of "r"
        # learn _SigmoidCalibration "a" and "b"
        # optimize _SigmoidCalibration with brier_score_loss (from sklearn.metrics import brier_score_loss)
        self.sig = _SigmoidCalibration().fit(X, y)
        
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        self.r = self.sig.predict(X)

        # if pi = Max({p1,..., pk})
        max_idx = np.argmax(X)
        X[max_idx] = X[max_idx] + self.r * (1-X[max_idx])
        # otherwise
        X[:max_idx] = X[:max_idx] * (1-self.r)
        X[max_idx+1:] = X[max_idx+1:] * (1-self.r)

        self.X_ = X

        return self.X_