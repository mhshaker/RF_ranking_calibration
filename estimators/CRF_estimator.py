import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.calibration import _SigmoidCalibration
from CalibrationM import convert_prob_2D
from sklearn.metrics import brier_score_loss

class CRF_calib(BaseEstimator, ClassifierMixin):
    
    def __init__(self, r=0.5, r_step=0.1, learning_method='brier_opt'):
        
        self.r = r
        self.r_step = r_step
        self.learning_method = learning_method


    def fit(self, X, y): # X,y are calibration dataset

        # [Goal] set the value of "r"
        # learn _SigmoidCalibration "a" and "b"
        # optimize _SigmoidCalibration with brier_score_loss (from sklearn.metrics import brier_score_loss)
        if self.learning_method == 'brier_opt':
            low = 0
            up = 1
            r_list = np.arange(low, up + self.r_step, self.r_step)

            r_opt = 0
            min_brier_score = 10000000
            for r in r_list:
                y_p = self.predict(X,r)
                brier_score = brier_score_loss(y, y_p[:,1])
                if brier_score < min_brier_score:
                    min_brier_score = brier_score
                    r_opt = r
            self.r = r_opt

        else:
            self.sig = _SigmoidCalibration().fit(X, y)
        
        return self


    def predict(self, X, r=-1):

        if self.learning_method == 'sig':
            r = self.sig.predict(X)
        elif self.learning_method == 'brier_opt' and r==-1:
            r = self.r

        X_2d = convert_prob_2D(X)        

        max_idx = np.argmax(X_2d, axis=1)
        min_idx = np.argmin(X_2d, axis=1)

        max_indices = list(range(len(X_2d))), max_idx
        min_indices = list(range(len(X_2d))), min_idx

        X_2d[max_indices] += r * (1- X_2d[max_indices])
        X_2d[min_indices] *= (1-r)

        y = X_2d

        return y