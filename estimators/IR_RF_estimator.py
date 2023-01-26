import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble._forest import ForestClassifier


def tree_laplace_corr(tree, x_data, laplace_smoothing, return_counts=False):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            if return_counts:
                tree_prob[data_index][i] = v + L
            else:
                tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class IR_RF(RandomForestClassifier):

    # refrence_prob = np.array()

    def predict_proba(self, X, laplace=1, return_tree_prob=False):
        prob_matrix  = []
        for estimator in self.estimators_:
            tree_prob = tree_laplace_corr(estimator,X, laplace)
            prob_matrix.append(tree_prob)
        prob_matrix = np.array(prob_matrix)
        prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
        if return_tree_prob:
            return prob_matrix
        else:
            return prob_matrix.mean(axis=1)

    def tree_leafcounts(self, X, laplace=1):
        leafcount_matrix  = []
        for estimator in self.estimators_:
            tree_prob = tree_laplace_corr(estimator,X, laplace, return_counts=True)
            leafcount_matrix.append(tree_prob)
        leafcount_matrix = np.array(leafcount_matrix)
        leafcount_matrix = leafcount_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
        return leafcount_matrix

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def rank(self, X, class_to_rank=1, train_rank=False, return_tree_rankings=False):
        probs = self.predict_proba(X, laplace=1, return_tree_prob=True)[:,:,class_to_rank]
        # print("probs\n", probs.transpose([1,0]))
        # probs = self.tree_leafcounts(X, laplace=1)[:,:,class_to_rank] # not probs but the number of instances in the leafs of the trees
        prob_arg_sort = np.argsort(probs, axis=0, kind="stable")
        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        # print("sorted prob index\n", prob_arg_sort)
        # exit()
        probs = probs.transpose([1,0]) 
        prob_rank = probs.copy()
        if train_rank:
            self.refrence_prob = np.sort(probs, axis=1) # np.sort(probs, axis=0, kind="stable")
            # print("refrence_prob", self.refrence_prob)
            # exit()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)): # loop thrugh trees
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r
        # print("prob_rank", prob_rank.shape)
        IR_RF_rank = prob_rank.sum(axis=0)
        # print("---------------------------------")
        # print("IR_RF_rank", IR_RF_rank)
        # exit()

        if return_tree_rankings:
            return prob_rank
        else:
            return IR_RF_rank

    def rank_refrence(self, X, class_to_rank=1): # find where X fals in the ranking of calib data
        probs = self.predict_proba(X, laplace=1, return_tree_prob=True)[:,:,class_to_rank]
        ranks = []
        for prob in probs: # loop through data
            data_rank = 0
            for tree_prob, ref_prob in zip(prob, self.refrence_prob): # loop through trees
                data_rank += find_nearest(ref_prob, tree_prob)
            ranks.append(data_rank)
        return np.array(ranks)

    def rank_fa(self, X, class_to_rank=1): # rank fa
        favor = self.tree_leafcounts(X, laplace=1)[:,:,class_to_rank] # not favor but the number of instances in the leafs of the trees
        prob_arg_sort = np.argsort(favor, axis=0, kind="stable")

        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        favor = favor.transpose([1,0]) 
        prob_rank = favor.copy()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)):
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r

        IR_RF_rank_favor = prob_rank.sum(axis=0)

        against = self.tree_leafcounts(X, laplace=1)[:,:,class_to_rank-1] # not probs but the number of instances in the leafs of the trees
        prob_arg_sort = np.argsort(-against, axis=0, kind="stable")

        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        against = against.transpose([1,0]) 
        prob_rank = against.copy()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)):
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r

        IR_RF_rank_against = prob_rank.sum(axis=0)

        IR_RF_rank = IR_RF_rank_favor + np.flip(IR_RF_rank_against)
        return IR_RF_rank

