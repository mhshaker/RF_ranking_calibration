from sklearn.isotonic import IsotonicRegression
import numpy as np
from estimators.IR_RF_estimator import IR_RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_classification
import Data.data_provider as dp
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from sklearn.calibration import _SigmoidCalibration

# data
X, y = make_classification(n_samples=100000, n_features=40, n_informative=2, n_redundant=10, random_state=42)
# X, y = dp.load_data("spambase")

x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=0)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=0) 


irrf = IR_RF(n_estimators=100, random_state=0)
irrf.fit(x_train, y_train)

rf_p_calib = irrf.predict_proba(x_calib, laplace=1)
rf_p_test = irrf.predict_proba(x_test, laplace=1)

iso_rf = IsotonicRegression().fit(rf_p_calib[:,1], y_calib)
rf_cp_test = iso_rf.predict(rf_p_test[:,1])


x_calib_rank = irrf.rank(x_calib, class_to_rank=1)
x_test_rank = irrf.rank(x_test, class_to_rank=1)

iso = IsotonicRegression(out_of_bounds = 'clip').fit(x_calib_rank, y_calib)
irrf_cp_test = iso.predict(x_test_rank)

# sig = _SigmoidCalibration().fit(x_calib_rank, y_calib)
# irrf_cp_test_sig = iso.predict(x_test_rank)

fop, mpv = calibration_curve(y_test, rf_p_test[:,1], n_bins=10)
fop_iso, mpv_iso = calibration_curve(y_test, rf_cp_test, n_bins=10)
fop_irrf, mpv_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=10)
# fop_irrf_sig, mpv_irrf_sig = calibration_curve(y_test, irrf_cp_test_sig, n_bins=10)


# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv, fop, marker='.', label="RF")
plt.plot(fop_iso, mpv_iso, marker='.', label="RF+iso")
plt.plot(fop_irrf, mpv_irrf, marker='.', label="RF+rank+ios", c="black")
# plt.plot(fop_irrf_sig, mpv_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
plt.legend()
plt.savefig("calib_plot.png")