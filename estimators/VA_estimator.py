import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from CalibrationM import convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression


class VA_calib(BaseEstimator, ClassifierMixin):
    
    def fit(self, X, y):

        self.dataset = list(zip(X,y))
        return self
    

    def predict(self, X):

        p0, p1 = [], []

        for x in X:

            dataset0 = self.dataset + [(x,0)]
            # print("len dataset0", dataset0[1])
            iso0 = IsotonicRegression().fit(*zip(*dataset0))
            p0.append(iso0.predict([x]))
            
            dataset1 = self.dataset + [(x,1)]
            iso1 = IsotonicRegression().fit(*zip(*dataset1))
            p1.append(iso1.predict([x]))

        p0, p1 = np.array(p0).flatten(),np.array(p1).flatten()

        interval = np.concatenate((p0.reshape(-1,1), p1.reshape(-1,1)), axis=1)
        res = np.mean(interval, axis=1)
        return res