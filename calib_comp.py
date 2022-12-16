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
X_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.6, shuffle=True, random_state=0)
_, _, y_train_valid_r, y_test_r = train_test_split(X, y_r, test_size=0.6, shuffle=True, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.5, shuffle=True, random_state=0) 
_, _, y_test_r, y_calib_r = train_test_split(X_train_valid, y_train_valid_r, test_size=0.5, shuffle=True, random_state=0) 


clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)

clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
cal_clf.fit(X_valid, y_valid)









# # Create classifiers
# lr = LogisticRegression()
# gnb = GaussianNB()
# rfc = RandomForestClassifier()
# lr_rf = LR_RF()

# clf_list = [
#     (lr, "Logistic"),
#     (gnb, "Naive Bayes"),
#     (rfc, "Random forest"),
#     (rfc, "LR_RF"),
# ]

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# fig = plt.figure(figsize=(10, 10))
# gs = GridSpec(4, 2)
# colors = plt.cm.get_cmap("Dark2")

# ax_calibration_curve = fig.add_subplot(gs[:2, :2])
# calibration_displays = {}
# markers = ["^", "v", "s", "o"]
# for i, (clf, name) in enumerate(clf_list):
#     if name == "LR_RF":
#         clf.fit(X_train, y_train, y_train_rank)
#     else:
#         clf.fit(X_train, y_train)
#     display = CalibrationDisplay.from_estimator(
#         clf,
#         X_test,
#         y_test,
#         n_bins=10,
#         name=name,
#         ax=ax_calibration_curve,
#         color=colors(i),
#         marker=markers[i],
#     )
#     calibration_displays[name] = display

# ax_calibration_curve.grid()
# ax_calibration_curve.set_title("Calibration plots")

# # Add histogram
# grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
# for i, (_, name) in enumerate(clf_list):
#     row, col = grid_positions[i]
#     ax = fig.add_subplot(gs[row, col])

#     ax.hist(
#         calibration_displays[name].y_prob,
#         range=(0, 1),
#         bins=10,
#         label=name,
#         color=colors(i),
#     )
#     ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

# plt.tight_layout()
# plt.savefig("calib_comp.png")


