from sklearn.ensemble import RandomForestClassifier
import Data.data_provider as dp
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0, oob_score=True)
clf.fit(X, y)
oob = clf.oob_decision_function_
print("oob", oob)
print("x", X.shape)
print("oob", oob.shape)