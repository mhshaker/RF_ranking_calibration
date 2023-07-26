from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import Data.data_provider as dp


n_features = 2
n_samples = 1500
n_copy = 50

X, _ = make_classification(n_samples=n_samples, 
                            n_features=2, 
                            n_informative=2, 
                            n_redundant=0, 
                            n_repeated=0, 
                            n_classes=2, 
                            n_clusters_per_class=2, 
                            weights=None, 
                            flip_y=0.05, 
                            class_sep=1.0, 
                            hypercube=True, 
                            shift=0.0, 
                            scale=1.0, 
                            shuffle=True, 
                            random_state=0)
XX, yy, PP = dp.x_y_q(X, n_copy)



x_train, x_test, y_train, y_test = train_test_split(XX, yy, test_size=0.1, shuffle=True, random_state=0)
_, _, tp_train, tp_test = train_test_split(XX, PP, test_size=0.1, shuffle=True, random_state=0)

rf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
# rf = DecisionTreeClassifier().fit(x_train, y_train)
# print("rf score", rf.score(x_test,y_test))

s = rf.predict_proba(XX)

# calculate C_test
u, counts = np.unique(s[:,1], return_counts=True)

# print(f">>>> u {u} counts {counts}")

c = PP.copy()
c[:] = 0
for v in u:
    e_index = np.argwhere(s[:,1] == v)
    e_labels = PP[e_index]
    e_labels_mean = e_labels.mean()
    # print("y_test[e_index]", y_test[e_index])
    # print("e_index", e_index)
    # print(">>>> c", e_labels_mean)
    c[e_index] = e_labels_mean
#     print(">>>>>>> c", c)
#     print("---------------------------------")
# print("---------------------------------")
# print("c", c) 

# dummy_clf = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
# print("du score", dummy_clf.score(x_test,y_test))f

from sklearn.metrics import mean_squared_error

y = yy
q = PP
s = s[:,1] 

# y = [1,1,1,0,1,1,0,0]
# q = [1,1,0.5,0.5,0.5,0.5,0.5,0.5]
# s = [0.9,0.9,0.9,0.9,0.4,0.4,0.4,0.4]
# c = [0.75,0.75,0.75,0.75,0.5,0.5,0.5,0.5]

# print("y", y)
# print("s", s)
# print("c", c)
# print("q", q)

BS = mean_squared_error(s,y)
CL = mean_squared_error(s,c)
GL = mean_squared_error(c,q)
IL = mean_squared_error(q,y)

print("---------------------------------")
print("BS", BS)
print("BS_d", CL + GL + IL)
