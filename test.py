import Data.data_provider as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

data_name = "spambase"

X, y = dp.load_data(data_name, "./")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

rf = RandomForestClassifier(random_state=0)

# search_space = {
#     "criterion": ["gini", "entropy"],
# }


search_space = {
    "n_estimators": [100],
    "max_depth": [15, 20, 25],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2,3,4,5],
    "min_samples_leaf": [1,2,3],
}

# GS = GridSearchCV(estimator=rf, param_grid=search_space, scoring=["accuracy"], refit="accuracy", cv=5)
# GS.fit(x_train, y_train)

RS = RandomizedSearchCV(rf, search_space, scoring=["accuracy"], refit="accuracy", cv=5, n_iter=10, random_state=0)
RS.fit(x_train, y_train)

# print("GS best", GS.best_params_)
print("RS best", RS.best_params_)

# rf_best = GS.best_estimator_
# print("acc", rf_best.score(x_test, y_test))