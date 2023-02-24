import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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

        for arr in X:
            # if pi = Max({p1,..., pk})
            max_idx = np.argmax(arr)
            arr[max_idx] = arr[max_idx] + self.r * (1-arr[max_idx])
            # otherwise
            arr[1-max_idx] = arr[1-max_idx] * (1-self.r)

        y = X

        return y