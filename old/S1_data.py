import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

seed = 0

calib = True
kendal = False

np.random.seed(seed)
# Synthetic data with 5 dimentions and 2 classes
samples = 10000
n_features = 5

mean1 = np.random.uniform(-5,5,n_features) #[0, 2, 3, -1, 9]
cov1 = np.zeros((n_features,n_features))
np.fill_diagonal(cov1, np.random.uniform(0,1,n_features))

# cov1 = [[.1, 0, 0, 0, 0], 
#         [0, .5, 0, 0, 0],
#         [0, 0, 0.8, 0, 0],
#         [0, 0, 0, .1, 0],
#         [0, 0, 0, 0, .3],
#         ]

mean2 = np.random.uniform(-5,5,n_features) # [-1, 3, 0, 2, 3]
cov2 = np.zeros((n_features,n_features))
np.fill_diagonal(cov2, np.random.uniform(0,1,n_features))

# cov2 = [[.9, 0, 0, 0, 0], 
#         [0, .1, 0, 0, 0],
#         [0, 0, 0.3, 0, 0],
#         [0, 0, 0, .1, 0],
#         [0, 0, 0, 0, .7],
#         ]

x1 = np.random.multivariate_normal(mean1, cov1, samples)
x2 = np.random.multivariate_normal(mean2, cov2, samples)

x1_pdf_dif = multivariate_normal.pdf(x1, mean1, cov1) - multivariate_normal.pdf(x1, mean2, cov2)
x2_pdf_dif = multivariate_normal.pdf(x2, mean2, cov2) - multivariate_normal.pdf(x2, mean1, cov1)

X = np.concatenate([x1, x2])
y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
tp = np.concatenate([x1_pdf_dif, x2_pdf_dif])


test_size  = 0.4
x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.5, shuffle=True, random_state=seed) 
_, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
_, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=0.5, shuffle=True, random_state=seed) 

irrf = IR_RF(n_estimators=100, random_state=seed)
irrf.fit(x_train, y_train)

if kendal:
        x_test_rank = irrf.rank(x_test, class_to_rank=1)
        x_test_rank_tree = irrf.rank(x_test, class_to_rank=1, return_tree_rankings=True)[0]
        print("x_test_rank_tree", x_test_rank_tree.shape)
        print("x_test_rank_tree", x_test_rank.shape)

        rank_sort_index = np.argsort(x_test_rank, kind="stable")
        true_sort_index = np.argsort(tp_test, kind="stable")

        y_test_rank_sort = y_test[rank_sort_index]
        y_test_true_sort = y_test[true_sort_index]

        tau, p_value = kendalltau(y_test_true_sort, y_test_rank_sort)
        print("tau", tau)

if calib:

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
        # plt.plot(fop_iso, mpv_iso, marker='.', label="RF+iso")
        # plt.plot(fop_irrf, mpv_irrf, marker='.', label="RF+rank+ios", c="black")
        # plt.plot(fop_true, mpv_true, marker='.', label="RF+true+ios", c="red")
        # plt.plot(fop_irrf_sig, mpv_irrf_sig, marker='.', label="RF+rank+sig", c="blue")
        plt.legend()
        plt.show()
        # plt.savefig("calib_plot.png")