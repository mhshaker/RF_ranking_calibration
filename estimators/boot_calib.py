from estimators.IR_RF_estimator import IR_RF
import numpy as np
from sklearn.calibration import calibration_curve

def find_nearest_index(arr, X):
    sorted_arr = np.sort(arr)
    absolute_diff = np.abs(sorted_arr - X)
    nearest_index = np.argmin(absolute_diff)
    return nearest_index


class Boot_calib():

    def __init__(self, bootstrap_size=100, boot_count=100, bins=30):

        self.bootstrap_size = bootstrap_size
        self.boot_count = boot_count
        self.bins = bins

    def fit(self, x_train, y_train, model):
        ens = []
        p_all = np.zeros(1)
        y_all = np.zeros(1)
        for boot_index in range(self.boot_count):
            model.random_state = boot_index * 100 # change the random seed to fit again
            # model = IR_RF(n_estimators=10  , oob_score=False, max_depth= 6, random_state=boot_index) # changing the RF params
            model.fit(x_train, y_train)
            p = model.predict_proba(x_train)
            if len(p_all) == 0:
                p_all = p
                y_all = y_train
            else:
                p_all = np.concatenate((p[:,1], p_all))
                y_all = np.concatenate((y_train, y_all))
            ens.append(p.copy())

        bin = int(len(p_all)/20)
        if bin > 100:
            bin = 100
        if bin < 10:
            bin = 10
        self.prob_true, self.prob_pred = calibration_curve(y_all, p_all, n_bins=bin)
        return self

    def predict_boot(self, X):

        b = []
        for boot_index in range(self.boot_count):
            tree_index = np.random.randint(low=0, high=X.shape[1], size=self.bootstrap_size)
            boot = X[:,tree_index,:]
            b.append(boot.copy())
        b = np.array(b)
        b = np.mean(b, axis=2) # average trees in each bootstrap
        b = np.mean(b, axis=0) # average all the bootstraps

        return b

    def predict_ens(self, x_test, x_train, y_train, model, param_change=False):

        ens = []
        params = model.get_params().copy()
        params_org = model.get_params().copy()

        depth_extention = np.random.randint(low=-1, high=1, size=self.boot_count)

        for boot_index, de in zip(range(self.boot_count), depth_extention):
            model.random_state = boot_index * 100 # change the random seed to fit again
            if param_change:
                model.max_depth = params['max_depth'] + de
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the ens

        model.set_params(**params_org)

        return b


    def predict_largeRF(self, x_test, x_train, y_train, model):

        ens = []
        params = model.get_params().copy()
        params_org = model.get_params().copy()

        model.n_estimators = params['n_estimators'] * self.boot_count

        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)

        model.set_params(**params_org)
        return p


    def predict_ensbin(self, x_test, x_train, y_train, model):

        ens = []
        for boot_index in range(self.boot_count):
            model.random_state = boot_index * 100 # change the random seed to fit again
            # model = IR_RF(n_estimators=10  , oob_score=False, max_depth= 6, random_state=boot_index) # changing the RF params
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the ens

        return b

    def predict_ens2(self, X):
        calib_prob = []
        for x in X:
            index = find_nearest_index(self.prob_pred, x)
            if index + 1 < len(self.prob_pred):
                p = (self.prob_true[index] + self.prob_true[index+1]) / 2
            else:
                p = self.prob_true[index]
            calib_prob.append(p)
        calib_prob = np.array(calib_prob)
        return calib_prob