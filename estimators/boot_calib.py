from sklearn.utils import resample
import numpy as np

class Boot_calib():

    def __init__(self, bootstrap_size=2, boot_count=40):

        self.bootstrap_size = bootstrap_size
        self.boot_count = boot_count

    def predict(self, X):
        print("X", X.shape)

        b = []
        for boot_index in range(self.boot_count):
            tree_index = np.random.randint(low=0, high=X.shape[1], size=self.bootstrap_size)
            boot = X[:,tree_index,:]
            b.append(boot.copy())
        b = np.array(b)
        print("b shape", b.shape)
        b = np.mean(b, axis=2)
        print("b shape", b.shape)
        b = np.mean(b, axis=0)
        print("b shape", b.shape)

        # p = [] #np.array(probs)
        # for data_point in X:
        #     d_p = []
        #     for sampling_seed in range(self.bootstrap_size):
        #         d_p.append(resample(data_point, random_state=sampling_seed))
        #         # print("d_p shape", d_p.shape)
        #     p.append(np.array(d_p))
        # p = np.array(p)
        # print("boot ", p.shape)
        # p = np.mean(p, axis=1) # average bootstraps
        # p = np.mean(p, axis=1) # average trees

        # print("boot mean", p.shape)
        return b