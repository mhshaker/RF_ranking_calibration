import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.calibration import _SigmoidCalibration
from CalibrationM import convert_prob_2D

class CRF_calib(BaseEstimator, ClassifierMixin):

    def fit(self, X, y): # X,y are calibration dataset

        # [Goal] set the value of "r"
        # learn _SigmoidCalibration "a" and "b"
        # optimize _SigmoidCalibration with brier_score_loss (from sklearn.metrics import brier_score_loss)
        self.sig = _SigmoidCalibration().fit(X, y)
        
        return self

    def predict(self, X):

        X_2d = convert_prob_2D(X)
        # Check if fit has been called
        # check_is_fitted(self)

        self.r = self.sig.predict(X)

        max_idx = np.argmax(arr, axis=1)
        # arg min
        X_2d[max_idx] += self.r * (1-arr[max_idx])
        X_2d[min_inx] += arr[1-max_idx] = arr[1-max_idx] * (1-self.r)
        
        for arr in X_2d:
            # if pi = Max({p1,..., pk})
            max_idx = np.argmax(arr)
            arr[max_idx] = arr[max_idx] + self.r * (1-arr[max_idx])
            # otherwise
            arr[1-max_idx] = arr[1-max_idx] * (1-self.r)

        y = X_2d

        return y