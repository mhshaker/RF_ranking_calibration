import numpy as np

import Data.data_provider as dp
import Experiments.core_exp as exp


def test_load():
    x, y = dp.load_data("spambase")
    assert len(x) > 0

def test_data_runs_cv():
    params = {
        "seed": 0,
        "runs": 2,
        "path": ".",
        "exp_name": "split_test",
        "split": "CV", #CV و random_split
        "cv_folds": 3,
        "plot_data": False,
        "data_name": "synthetic",
        "data_size": 10,
        "n_features": 20,
        "class1_mean_min":0, 
        "class1_mean_max":1,
        "class2_mean_min":1, 
        "class2_mean_max":3, 
    }

    exp_data_name = "test_data_runs"
    data_runs = exp.load_data_runs(params, exp_data_name)
    assert len(data_runs) == 6 # number of data : 2 runs of CV with 3 folds is 6 data

def test_data_runs_random_split():
    params = {
        "seed": 0,
        "runs": 5,
        "path": ".",
        "exp_name": "split_test",
        "split": "random_split", #CV و random_split
        "test_split": 0.3,
        "calib_split": 0.1,
        "plot_data": False,
        "data_name": "synthetic",
        "data_size": 10,
        "n_features": 20,
        "class1_mean_min":0, 
        "class1_mean_max":1,
        "class2_mean_min":1, 
        "class2_mean_max":3, 

    }

    exp_data_name = "test_data_runs"
    data_runs = exp.load_data_runs(params, exp_data_name)
    assert len(data_runs) == 5 # number of data: 5 runs with random_split is 5 data

def test_data_synthetic_parts(): # test to see if a data dict consists of all the parts that we require from it such as x_train, x_test, ...
    params = {
        "seed": 0,
        "runs": 5,
        "path": ".",
        "exp_name": "split_test",
        "split": "random_split", #CV و random_split
        "test_split": 0.3,
        "calib_split": 0.1,
        "plot_data": False,
        "data_name": "synthetic",
        "data_size": 10,
        "n_features": 20,
        "class1_mean_min":0, 
        "class1_mean_max":1,
        "class2_mean_min":1, 
        "class2_mean_max":3, 

    }

    exp_data_name = "test_data_runs"
    data_runs = exp.load_data_runs(params, exp_data_name)

    expected_keys = ['name', 'X', 'y', 'x_train_calib', 'x_test', 'y_train_calib', 'y_test', 'x_train', 'x_calib', 'y_train', 'y_calib', 'tp', 'tp_train_calib', 'tp_test', 'tp_train', 'tp_calib'] # keys that are required to be in a data
    data_kyes = list(data_runs[0].keys())

    expected_keys.sort()
    data_kyes.sort()

    assert data_kyes == expected_keys

def test_data_real(): # test to see if a data dict consists of all the parts that we require from it such as x_train, x_test, ...
    params = {
        "seed": 0,
        "runs": 3,
        "path": ".",
        "exp_name": "split_test",
        "split": "CV", #CV و random_split
        "cv_folds": 10,
        "plot_data": False,
        "data_name": "parkinsons",
    }

    exp_data_name = "test_data_runs"
    data_runs = exp.load_data_runs(params, exp_data_name, ".")

    expected_keys = ['name', 'X', 'y', 'x_train_calib', 'x_test', 'y_train_calib', 'y_test', 'x_train', 'x_calib', 'y_train', 'y_calib'] # keys that are required to be in a data
    data_kyes = list(data_runs[0].keys())

    expected_keys.sort()
    data_kyes.sort()

    print("expected_keys", expected_keys)
    print("data_kyes", data_kyes)

    assert data_kyes == expected_keys
    assert len(data_runs) == 30
    assert len(np.unique(data_runs[0]["y_calib"])) > 1 
    assert len(np.unique(data_runs[0]["y_test"])) > 1
    assert len(np.unique(data_runs[0]["y_train"])) > 1

