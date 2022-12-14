# connect to DB and load datasets
# calibration methods to run on datasets
# calibration measures for evaluation

### road map

# implement LR-RF
    # convert normal data to ranking data
    # train RF on normal data
    # at test time fetch the lables of each leaf from ranking data
    # agregate the ranking labels using Borda method
# convert ranking to calibration using ISO
# compare to other calib approches

###

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# import Data.data_provider as dp
# dataset = "adult"
# features, target = dp.load_data(dataset)

def create_neighbors_ranking(tree, x_train, y_train_rank):
    tree_borda_dict = {}
    # print("y_train_rank")
    # print(y_train_rank)
    leaf_index_array = tree.apply(x_train)
    leaf_nodes = np.unique(leaf_index_array, return_counts=False)
    # print("leaf_index_array ", leaf_index_array)
    # print("unique leaf_nodes ", leaf_nodes)
    for leaf_node in leaf_nodes:
        leaf_nodes_borda = y_train_rank[np.argwhere(leaf_index_array==leaf_node)].sum(axis=0)
        tree_borda_dict[leaf_node] = np.array(leaf_nodes_borda).reshape(-1)
        # print(f"leaf_node {leaf_node} borda ", leaf_nodes_borda)
    # print(np.unique(leaf_index_array, return_counts=False))
    # plt.figure(figsize=(20, 20))
    # plot_tree(tree)
    # plt.savefig(f"tree{i}.png",format='png',bbox_inches = "tight")

    return tree_borda_dict

def get_ranking(tree, x_test, borda):
    leaf_index_array = tree.apply(x_test)
    rankings = []
    for leaf_index in leaf_index_array:
        rankings.append(borda[leaf_index]) 
    rankings = np.array(rankings)
    return rankings

# load the data
from sklearn.datasets import load_digits
digits = load_digits()
features, target = digits.data, digits.target

# convert lables to ranking
gnb = GaussianNB()
y_pred = gnb.fit(features, target).predict_proba(features)
y_argsort = np.argsort(y_pred, axis=1, kind="stable")
target_rank = y_pred.copy()
for index, (y_r, y_a) in enumerate(zip(target_rank, y_argsort)):
    for i in range(len(y_r)):
        y_r[y_a[i]] = i
    target_rank[index] = y_r

# print(y_pred[2])
# print(target_rank[2])
# exit()

# train RF
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=0)
_, _, y_train_rank, y_test_rank = train_test_split(features, target_rank, test_size=0.5, random_state=0)

model = RandomForestClassifier(max_depth=2, n_estimators=3, random_state=0)
model.fit(x_train, y_train)

# get neighbor rankings
borda_dict_list = []
for i, estimator in enumerate(model.estimators_):
    borda_dict_list.append(create_neighbors_ranking(estimator,x_train, y_train_rank))

# first level aggregation
tree_rankings = []
for i, (estimator, borda) in enumerate(zip(model.estimators_, borda_dict_list)):
    tree_rankings.append(get_ranking(estimator,x_test,borda))

    print(estimator.tree_)
tree_rankings = np.array(tree_rankings)

# second level aggregation
RF_rankings = tree_rankings.sum(axis=0)

# convert ranking to prob
ranking_sum = RF_rankings.sum(axis=1)
LR_RF_probs = RF_rankings / ranking_sum[:, np.newaxis]

RF_probs = model.predict_proba(x_test)

print(RF_probs[5])
print(LR_RF_probs[5])

print(np.argmax(RF_probs[5]))
print(np.argmax(LR_RF_probs[5]))
print("--------------------------------")
print(y_test[5])











# def tree_laplace_corr(tree, x_data, laplace_smoothing, a=0, b=0):
#     tree_prob = tree.predict_proba(x_data)
#     leaf_index_array = tree.apply(x_data)
#     for data_index, leaf_index in enumerate(leaf_index_array):
#         leaf_values = tree.tree_.value[leaf_index]
#         leaf_samples = np.array(leaf_values).sum()
#         for i,v in enumerate(leaf_values[0]):
#             L = laplace_smoothing
#             if a != 0 or b != 0:
#                 if i==0:
#                     L = a
#                 else:
#                     L = b
#             # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
#             tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
#     return tree_prob