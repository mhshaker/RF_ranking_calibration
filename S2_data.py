import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
from sklearn.datasets import make_regression
from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

### Synthetic data generation
samples = 100000

X, tp = make_regression(samples) # make regression data
y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold

### spliting data to train calib and test
test_size = 0.4
x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
_, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
_, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

### training the IRRF
irrf = IR_RF(n_estimators=100, random_state=seed)
irrf.fit(x_train, y_train)

# ranking and getting prob for x_test
x_test_rank = irrf.rank(x_test, class_to_rank=1)
x_test_prob = irrf.predict_proba(x_test, laplace=1)[:, 1]

# sorting x_test based on ranking, prob, and true rankings
rank_sort_index = np.argsort(x_test_rank, kind="stable")
prob_sort_index = np.argsort(x_test_prob, kind="stable")
true_sort_index = np.argsort(tp_test, kind="stable")

tp_test_rank_sort = tp_test[rank_sort_index]
tp_test_prob_sort = tp_test[prob_sort_index]
tp_test_true_sort = tp_test[true_sort_index]

### ranking performance of the IRRF compared to true rankings
tau, p_value = kendalltau(tp_test_true_sort, tp_test_rank_sort)
print("tau", tau)

tau, p_value = kendalltau(tp_test_true_sort, tp_test_prob_sort)
print("tau", tau)


### calibration and ECE plot
rf_p_calib = irrf.predict_proba(x_calib, laplace=0)
rf_p_test = irrf.predict_proba(x_test, laplace=0)

iso_rf = IsotonicRegression().fit(rf_p_calib[:,1], y_calib)
rf_cp_test = iso_rf.predict(rf_p_test[:,1])


x_calib_rank = irrf.rank(x_calib, class_to_rank=1)
x_test_rank = irrf.rank(x_test, class_to_rank=1)

iso = IsotonicRegression().fit(x_calib_rank, y_calib)
irrf_cp_test = iso.predict(x_test_rank)

iso_true = IsotonicRegression().fit(tp_calib, y_calib)
true_cp_test = iso_true.predict(tp_test)


# sig = _SigmoidCalibration().fit(x_calib_rank, y_calib)
# irrf_cp_test_sig = iso.predict(x_test_rank)

fop, mpv = calibration_curve(y_test, rf_p_test[:,1], n_bins=10)
fop_iso, mpv_iso = calibration_curve(y_test, rf_cp_test, n_bins=10)
fop_irrf, mpv_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=10)
fop_true, mpv_true = calibration_curve(y_test, true_cp_test, n_bins=10)
# fop_irrf_sig, mpv_irrf_sig = calibration_curve(y_test, irrf_cp_test_sig, n_bins=10)


# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.', label="RF")
plt.plot(fop_iso, mpv_iso, marker='.', label="RF+iso")
plt.plot(fop_irrf, mpv_irrf, marker='.', label="RF+rank+ios", c="black")
plt.plot(fop_true, mpv_true, marker='.', label="RF+true+ios", c="red")
# plt.plot(fop_irrf_sig, mpv_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
plt.legend()
plt.show()
# plt.savefig("calib_plot.png")