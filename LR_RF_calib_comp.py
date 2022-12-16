import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from LR_RF_estimator import LR_RF
from LR_RF_estimator import convert_to_ranking
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV


X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42, cluster_std=5.0)

# convert lables to ranking
y_r, y_rt = convert_to_ranking(X, y)

# split to train calib and test
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.6, shuffle=True, random_state=0)
_, _, y_train_valid_r, y_test_r = train_test_split(X, y_r, test_size=0.6, shuffle=True, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.5, shuffle=True, random_state=0) 
_, _, y_train_r, y_valid_r = train_test_split(X_train_valid, y_train_valid_r, test_size=0.5, shuffle=True, random_state=0) 


clf = RandomForestClassifier(n_estimators=25, random_state=0)
clf.fit(X_train_valid, y_train_valid)

clf = RandomForestClassifier(n_estimators=25, random_state=0)
clf.fit(X_train, y_train)
cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
cal_clf.fit(X_valid, y_valid)

model_lr = LR_RF(n_estimators=25, random_state=0)
model_lr.fit(X_train, y_train, y_train_r)
cal_model_lr = CalibratedClassifierCV(model_lr, method="sigmoid", cv="prefit")
cal_model_lr.fit(X_valid, y_valid)

# get probs from all the models
clf_probs = clf.predict_proba(X_test) # RF
cal_clf_probs = cal_clf.predict_proba(X_test) # RF calib

model_lr_plobs = model_lr.predict_proba(X_test) # LR_RF 
cal_model_lr_plobs = cal_model_lr.predict_proba(X_test) # LR_RF calib

# log loss metric
score = log_loss(y_test, clf_probs)
cal_score = log_loss(y_test, cal_clf_probs)
score_LF_RF = log_loss(y_test, model_lr_plobs)
cal_score_LF_RF = log_loss(y_test, cal_model_lr_plobs)

print("Log-loss of")
print(f" * uncalibrated RF: {score:.3f}")
print(f" * calibrated RF: {cal_score:.3f}")
print(f" * uncalibrated LR_RF: {score_LF_RF:.3f}")
print(f" * calibrated LR_RF: {cal_score_LF_RF:.3f}")


# plot changes in prob after calibration

import matplotlib.pyplot as plt


for probs, cal_probs, name in zip([clf_probs, model_lr_plobs],[cal_clf_probs, cal_model_lr_plobs], ["RF", "LR_RF"]):
    plt.figure(figsize=(10, 10))
    colors = ["r", "g", "b"]

    # Plot arrows
    for i in range(probs.shape[0]):
        plt.arrow(
            probs[i, 0],
            probs[i, 1],
            cal_probs[i, 0] - probs[i, 0],
            cal_probs[i, 1] - probs[i, 1],
            color=colors[y_test[i]],
            head_width=1e-2,
        )

    # Plot perfect predictions, at each vertex
    plt.plot([1.0], [0.0], "ro", ms=20, label="Class 1")
    plt.plot([0.0], [1.0], "go", ms=20, label="Class 2")
    plt.plot([0.0], [0.0], "bo", ms=20, label="Class 3")

    # Plot boundaries of unit simplex
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

    # Annotate points 6 points around the simplex, and mid point inside simplex
    plt.annotate(
        r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)",
        xy=(1.0 / 3, 1.0 / 3),
        xytext=(1.0 / 3, 0.23),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)
    plt.annotate(
        r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)",
        xy=(0.5, 0.0),
        xytext=(0.5, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)",
        xy=(0.0, 0.5),
        xytext=(0.1, 0.5),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)",
        xy=(0.5, 0.5),
        xytext=(0.6, 0.6),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $0$, $1$)",
        xy=(0, 0),
        xytext=(0.1, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($1$, $0$, $0$)",
        xy=(1, 0),
        xytext=(1, 0.1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.annotate(
        r"($0$, $1$, $0$)",
        xy=(0, 1),
        xytext=(0.1, 1),
        xycoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="center",
    )
    # Add grid
    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], "k", alpha=0.2)
        plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

    plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
    plt.xlabel("Probability class 1")
    plt.ylabel("Probability class 2")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    _ = plt.legend(loc="best")
    plt.savefig(f"plots/calib_comp_{name}.png")
    plt.close()


for model, name in zip([cal_clf, cal_model_lr], ["RF", "LR_RF"]):

    plt.figure(figsize=(10, 10))
    # Generate grid of probability values
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
    p = p[p[:, 2] >= 0]

    # Use the three class-wise calibrators to compute calibrated probabilities
    calibrated_classifier = model.calibrated_classifiers_[0]
    prediction = np.vstack(
        [
            calibrator.predict(this_p)
            for calibrator, this_p in zip(calibrated_classifier.calibrators, p.T)
        ]
    ).T

    # Re-normalize the calibrated predictions to make sure they stay inside the
    # simplex. This same renormalization step is performed internally by the
    # predict method of CalibratedClassifierCV on multiclass problems.
    prediction /= prediction.sum(axis=1)[:, None]

    # Plot changes in predicted probabilities induced by the calibrators
    for i in range(prediction.shape[0]):
        plt.arrow(
            p[i, 0],
            p[i, 1],
            prediction[i, 0] - p[i, 0],
            prediction[i, 1] - p[i, 1],
            head_width=1e-2,
            color=colors[np.argmax(p[i])],
        )

    # Plot the boundaries of the unit simplex
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], "k", alpha=0.2)
        plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

    plt.title("Learned sigmoid calibration map")
    plt.xlabel("Probability class 1")
    plt.ylabel("Probability class 2")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.savefig(f"plots/grid_{name}.png")
