import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from CalibrationM import convert_prob_2D
from sklearn.linear_model import LogisticRegression
from estimators.LR_estimator import LR_u as lr

class treeLR_calib(BaseEstimator, ClassifierMixin):

    remove_refrence_node = True
    retrain_alpha = True
    
    def convert_data_to_lr(self, tree, X, all_leafs=np.zeros(10), ref_index=-1, remove_refrence_node=False):
        leaf_index_array = tree.apply(X) # get leaf index for each data in X_train
        if all_leafs.all() == 0:
            all_leafs = np.array(range(tree.tree_.node_count)) # list of all leafs in the tree
        lr_x = []
        for leaf in all_leafs: # create dummy variables for each leaf
            dummy = np.where(leaf_index_array==leaf, 1, 0)
            lr_x.append(dummy)
        lr_x = np.array(lr_x)
        
        if remove_refrence_node:
            if ref_index == -1:        
                dummy_sum = lr_x.sum(axis=1)
                ref_index = dummy_sum.argmax()
            lr_x = np.delete(lr_x, ref_index, 0)
        lr_x = lr_x.transpose(1,0)
        return lr_x, all_leafs, ref_index


    def fit(self, RF, x_train, y_train, x_calib, y_calib):
        
        self.RF = RF
        self.lr_list = []
        self.tree_leaf_list = []
        self.ref_index = []

        for estimator in RF.estimators_:
            lr_x_train, leafs, ref = self.convert_data_to_lr(estimator, x_train)
            lr_x_calib, _, _ = self.convert_data_to_lr(estimator, x_calib)
            self.tree_leaf_list.append(leafs)
            self.ref_index.append(ref)
            # train LR with training data
            tlr = lr(random_state=0).fit(lr_x_train, y_train)
            # retrain alpha with calib data
            tlr = tlr.update_intercept(lr_x_calib, y_calib)
            self.lr_list.append(tlr)

        return self


    def predict(self, X):
        probs = []
        for tree, lr, leafs, ref in zip(self.RF.estimators_, self.lr_list, self.tree_leaf_list, self.ref_index):
            lr_x, _, _ = self.convert_data_to_lr(tree, X, leafs, ref)
            p_lr = lr.predict(lr_x)
            probs.append(p_lr)
        lr_probs = np.array(probs).transpose(1,0)
        RF_calib_probs = lr_probs.mean(axis=1)

        return convert_prob_2D(RF_calib_probs)