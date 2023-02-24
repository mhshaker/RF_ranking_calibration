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

        r = self.sig.predict(X)
        
        X_2d = convert_prob_2D(X)        

        max_idx = np.argmax(X_2d, axis=1)
        min_idx = np.argmin(X_2d, axis=1)

        max_indices = list(range(len(X_2d))), max_idx
        min_indices = list(range(len(X_2d))), min_idx

        X_2d[max_indices] += r * (1- X_2d[max_indices])
        X_2d[min_indices] *= (1-r)

        y = X_2d

        return y