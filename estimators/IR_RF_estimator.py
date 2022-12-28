import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble._forest import ForestClassifier


def tree_laplace_corr(tree, x_data, laplace_smoothing, a=0, b=0):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            if a != 0 or b != 0:
                if i==0:
                    L = a
                else:
                    L = b
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob


class IR_RF(RandomForestClassifier):

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

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def rank(self, X, class_to_rank=1, return_tree_rankings=False):
        probs = self.predict_proba(X, laplace=1, return_tree_prob=True)[:,:,class_to_rank]
        prob_arg_sort = np.argsort(probs, axis=0, kind="stable")

        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        probs = probs.transpose([1,0]) 
        prob_rank = probs.copy()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)):
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r


        IR_RF_rank = prob_rank.sum(axis=0)

        if return_tree_rankings:
            return prob_rank
        else:
            return IR_RF_rank


