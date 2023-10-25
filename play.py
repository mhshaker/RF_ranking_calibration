# Import necessary libraries
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import Data.data_provider as dp
import time


st = time.time()
# Load a sample dataset (e.g., the Iris dataset)
X, y = dp.load_data("datatrieve", './')
print("len x", len(X))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Gaussian Process Classifier with the kernel you want to use
kernel = 1.0 * RBF(length_scale=1.0)
gp_classifier = GaussianProcessClassifier(kernel=kernel)

# # Create a parameter grid for hyperparameter optimization
# param_grid = {
#     "kernel": [1.0 * RBF(length_scale=1.0), 1.0 * RBF(length_scale=0.5)],
#     "n_restarts_optimizer": [0, 1, 2, 3],
# }

# # Create a GridSearchCV object to optimize hyperparameters
# grid_search = RandomizedSearchCV(gp_classifier, param_grid, scoring=["neg_brier_score"], refit="neg_brier_score", cv=5, n_iter=2, random_state=0)

# # Fit the model and optimize hyperparameters
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# print("Best hyperparameters:", grid_search.best_params_)

# # Evaluate the model with the best hyperparameters on the test set
# best_model = grid_search.best_estimator_

gp_classifier.fit(X_train, y_train)
best_model = gp_classifier
accuracy = best_model.score(X_test, y_test)
print("Test accuracy:", accuracy)

et = time.time() - st

print("run time", et)