import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from CalibrationM import convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression


class VA_calib(BaseEstimator, ClassifierMixin):
    
    def predict(train, test):

        p0, p1 = [], []

        for x in test:
            train0 = train + [(x,0)]
            iso0 = IsotonicRegression().fit(*zip(*train0))
            p0.append(iso0.predict([x]))
            
            train1 = train + [(x,1)]
            iso1 = IsotonicRegression().fit(*zip(*train1))
            p1.append(iso1.predict([x]))

        flatenned_p0, flattened_p1 = np.array(p0).flatten(), np.array(p1).flatten()
        
        return flatenned_p0, flattened_p1
        # how to call
        # p0d,p1d = VennABERS_by_def(list(zip(xs,ys)), xtest)