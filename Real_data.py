
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
from sklearn.datasets import make_regression
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from CalibrationM import confidance_ECE, convert_prob_2D, classwise_ECE
import Data.data_provider as dp
from sklearn.calibration import _SigmoidCalibration
import matplotlib.pyplot as plt

runs = 1
n_estimators=100

plot_bins = 10
test_size = 0.3

calib = True
plot = True

ECE_rf_list = []
ECE_sig_list = []
ECE_iso_list = []
ECE_irrf_list = []

seed = 0
X, y = dp.load_data("spambase") # spambase climate QSAR

for min_samples_leaf in [2]:
    print("--------------------------------- min_samples_leaf", min_samples_leaf)
    for seed in range(runs):
        # seed = 5
        # print("seed ", seed)
        np.random.seed(seed)

        ### spliting data to train calib and test
        x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
        x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=0.6, shuffle=True, random_state=seed) 

        ### training the IRRF
        irrf = IR_RF(n_estimators=n_estimators, random_state=seed)
        irrf.fit(x_train, y_train)

        if calib:
            ### calibration and ECE plot

            # random forest probs
            rf_p_calib = irrf.predict_proba(x_calib, laplace=1)
            rf_p_test = irrf.predict_proba(x_test, laplace=1)

            # sigmoid calibration on RF
            sig_rf = _SigmoidCalibration().fit(rf_p_calib[:,1], y_calib)
            rf_cp_sig_test = sig_rf.predict(rf_p_test[:,1])

            # ISO calibration on RF
            iso_rf = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_calib)
            rf_cp_test = iso_rf.predict(rf_p_test[:,1])

            # Ranking with the RF
            x_calib_rank = irrf.rank(x_calib, class_to_rank=1, train_rank=True)
            x_test_rank = irrf.rank_refrence(x_test, class_to_rank=1)


            # RF ranking + ISO
            iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, y_calib) 
            irrf_cp_test = iso_rank.predict(x_test_rank)

            ece_rf = confidance_ECE(rf_p_test, y_test, bins=plot_bins)
            ece_sig = confidance_ECE(convert_prob_2D(rf_cp_sig_test), y_test, bins=plot_bins)
            ece_iso = confidance_ECE(convert_prob_2D(rf_cp_test), y_test, bins=plot_bins)
            ece_irrf = confidance_ECE(convert_prob_2D(irrf_cp_test), y_test, bins=plot_bins)

            # ece_rf = classwise_ECE(rf_p_test, y_test, bins=plot_bins, full_ece=True)
            # ece_sig = classwise_ECE(convert_prob_2D(rf_cp_sig_test), y_test, bins=plot_bins, full_ece=True)
            # ece_iso = classwise_ECE(convert_prob_2D(rf_cp_test), y_test, bins=plot_bins, full_ece=True)
            # ece_irrf = classwise_ECE(convert_prob_2D(irrf_cp_test), y_test, bins=plot_bins, full_ece=True)


            ECE_rf_list.append(ece_rf)
            ECE_sig_list.append(ece_sig)
            ECE_iso_list.append(ece_iso)
            ECE_irrf_list.append(ece_irrf)

            if plot:
                tp, pp = calibration_curve(y_test, rf_p_test[:,1], n_bins=plot_bins)
                tp_iso, pp_iso = calibration_curve(y_test, rf_cp_test, n_bins=plot_bins)
                tp_sig, pp_sig = calibration_curve(y_test, rf_cp_sig_test, n_bins=plot_bins)
                tp_irrf, pp_irrf = calibration_curve(y_test, irrf_cp_test, n_bins=plot_bins)
                



                plt.plot([0, 1], [0, 1], linestyle='--')
                plt.plot(tp, pp, marker='.', label="RF")
                # plt.plot(tp_sig, pp_sig, marker='.', label="RF+sig")
                plt.plot(tp_iso, pp_iso, marker='.', label="RF+iso")
                plt.plot(tp_irrf, pp_irrf, marker='.', label="RF+rank+ios", c="black")
                plt.xlabel("True probability")
                plt.ylabel("Mean predicted probability")
                plt.legend()
                plt.show()

                plt.hist(pp_iso, color="green")
                plt.ylabel("Count")
                plt.xlabel("Mean predicted probability")

                plt.show()
                plt.hist(pp,color="black")
                plt.ylabel("Count")
                plt.xlabel("Mean predicted probability")
                plt.show()



    if calib:
        print("normal ece   ", np.array(ECE_rf_list).mean())
        print("sig ece      ", np.array(ECE_sig_list).mean())
        print("iso ece      ", np.array(ECE_iso_list).mean())
        print("IRRF iso ece ", np.array(ECE_irrf_list).mean())
