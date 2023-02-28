from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from fuzzytree import FuzzyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import Data.data_provider as dp

X, y = dp.load_data("spambase") # spambase climate QSAR parkinsons vertebral
# X, y = make_moons(n_samples=300, noise=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_fuzz = FuzzyDecisionTreeClassifier().fit(X_train, y_train)
clf_sk = DecisionTreeClassifier().fit(X_train, y_train)


print(f"fuzzytree: {clf_fuzz.score(X_test, y_test)}")
print(f"  sklearn: {clf_sk.score(X_test, y_test)}")