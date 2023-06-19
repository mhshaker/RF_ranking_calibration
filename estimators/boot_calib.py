from estimators.IR_RF_estimator import IR_RF
import numpy as np

class Boot_calib():

    def __init__(self, bootstrap_size=100, boot_count=100):

        self.bootstrap_size = bootstrap_size
        self.boot_count = boot_count

    def predict(self, X):

        b = []
        for boot_index in range(self.boot_count):
            tree_index = np.random.randint(low=0, high=X.shape[1], size=self.bootstrap_size)
            boot = X[:,tree_index,:]
            b.append(boot.copy())
        b = np.array(b)
        b = np.mean(b, axis=2) # average trees in each bootstrap
        b = np.mean(b, axis=0) # average all the bootstraps

        return b

    def predict_ens(self, x_test, x_train, y_train, model):

        ens = []
        for boot_index in range(self.boot_count):
            model.random_state = boot_index * 100 # change the random seed to fit again
            # model = IR_RF(n_estimators=10  , oob_score=False, max_depth= 6, random_state=boot_index) # changing the RF params
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the bootstraps

        return b

