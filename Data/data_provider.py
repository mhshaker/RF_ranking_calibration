
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.io import arff
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
import random
from scipy.integrate import quad


def unpickle(file): # for reading the CIFAR dataset
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def load_data(data_name, path="."):   

	df = pd.read_csv(f'{path}/Data/{data_name}.csv')
	features = np.array(df.drop("Class", axis=1))
	target = np.array(df["Class"])

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	if "cifar10" == data_name or "mnist" in data_name:
		features = features.astype('float32')
		features /= 255

	if "digits" == data_name:
		features = features.astype('float32')
		features /= 16
	if "cifar10small" == data_name:
		features = features.astype('float32')
		features /= 256

	return features, target

def load_ood(data_name, split=0.3, calibration_set=True, seed=1):
	features, target = load_data(data_name)
	np.random.seed(seed)
	classes = np.unique(target)
	selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
	selected_id_index = np.argwhere(np.isin(target,selected_id)) # get index of all id instances
	selected_ood_index = np.argwhere(np.isin(target,selected_id,invert=True)) # get index of all not selected classes (OOD)

	target_id    = target[selected_id_index].reshape(-1)
	features_id  = features[selected_id_index].reshape(-1, features.shape[1])
	target_ood   = target[selected_ood_index].reshape(-1)
	features_ood = features[selected_ood_index].reshape(-1, features.shape[1])    

	x_train, x_test_id, y_train, y_test_id  = split_data(features_id, target_id,   split=split, seed=seed)
	_      , x_test_ood, _     , y_test_ood = split_data(features_ood, target_ood, split=split, seed=seed)

	if calibration_set:
		x_test_id, x_calib, y_test_id, y_calib = train_test_split(x_test_id, y_test_id, test_size=0.5, shuffle=True, random_state=1)
	else:
		x_calib, y_calib = 0, 0

	minlen = len(x_test_id)
	if len(x_test_ood) < minlen:
		minlen = len(x_test_ood)

	y_test_idoodmix = np.concatenate((np.zeros(minlen), np.ones(minlen)), axis=0)
	x_test_idoodmix = np.concatenate((x_test_id[:minlen], x_test_ood[:minlen]), axis=0)

	return x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood, x_test_idoodmix, y_test_idoodmix, x_calib, y_calib

def split_data(features, target, split, seed=1):
   x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=split, shuffle=True, random_state=seed, stratify=target)
   return x_train, x_test, y_train, y_test

def balance_dataset(df):
	# Separate majority and minority classes
	y = df.Class.unique()
	df_loss = df[df.Class==y[0]]
	df_win  = df[df.Class==y[1]]
	
	max_len = len(df_loss)
	if len(df_win) > max_len:
		max_len = len(df_win)
	# Upsample minority class
	df_upsampled_win = resample(df_win, 
                                replace=True,           # sample with replacement
                                n_samples=max_len,      # to match majority class
                                random_state=123)       # reproducible results
	 
	# Combine majority class with upsampled minority class
	df_balance = pd.concat([df_loss, df_upsampled_win])
	return df_balance

def load_arff_data(name="adult", convert_to_int=True, type="binary", log=True):
	dataset = arff.load(open(f'/home/mhshaker/projects/uncertainty/Data/{name}.arff', 'r'))
	df = pd.DataFrame(dataset['data'])
	df = df.sample(frac=1).reset_index(drop=True)
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if log:
		print(f"Data name = {name}")
		print(df.head())
		print(df['target'].value_counts())

	if convert_to_int:
		for column in df:
			df[column] = df[column].astype("category").cat.codes
	# print(df.head())
	# exit()
	features = df.drop("target", axis=1)
	# print(features)
	features = preprocessing.scale(features)
	# print(features)

	# features_names = list(features.columns)
	# target_names = ["class1","class2"]

	return features, df.target #features_names, target_names

def load_arff_2(data_name):
	data = arff.loadarff(f"./Data/{data_name}.arff")
	df = pd.DataFrame(data[0])
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if data_name == "MagicTelescope":
		df.drop("ID", axis=1, inplace=True)

	features = df.drop("target", axis=1)
	target = df.target

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	# print(features.head())

	return np.array(features), np.array(target)


def x_y_q(X, n_copy=50, seed=0): # create true probability with repeating X instances n_copy times with different labels assigned by a random choice with prob p drawn from uniform dirstribution
	np.random.seed(seed)
	n_features = X.shape[1]
	n_samples = len(X)

	P = np.random.uniform(0,1,n_samples)

	XX = []
	yy = []
	PP = []
	for x, p in zip(X, P):
		y_r = np.random.choice([0,1], n_copy, p=[1-p, p])
		x_r = np.full((n_copy,n_features), x)

		u , counts = np.unique(y_r, return_counts=True)
		# print(f"u {u} counts {counts}")
		if len(counts) > 1:
			e_p = float(counts[1] / (counts[1] + counts[0]))
		else:
			e_p = float(u[0])
		# print(f"e_p type {type(e_p)} e_p {e_p}")
		# print("---------------------------------")
		# p_r = np.full(n_copy, p)
		# print("p_r", p_r)

		p_r = np.full(n_copy, e_p)
		# print("e_p", p_r)
		# print("r\n", r)
		# print("y", y_r)
		# print("---------------------------------")
		yy.append(y_r)
		XX.append(x_r)
		PP.append(p_r)

	XX = np.array(XX).reshape(-1, n_features)
	yy = np.array(yy).reshape(-1)
	PP = np.array(PP).reshape(-1)

	return XX, yy, PP

def get_pre_x(n_features):

	
	# only for parameters 

	# "class1_mean_min":0, 
    # "class1_mean_max":1,
    # "class2_mean_min":2, 
    # "class2_mean_max":3, 

    # "class1_cov_min":1, 
    # "class1_cov_max":2,
    # "class2_cov_min":1, 
    # "class2_cov_max":2.5, 
	

	pre_x = {
		2: 0.0,
		4: 0.25,
		6: 0.47,
		8: 0.56,
		10: 0.7,
		12: 0.72,
		14: 0.74,
		16: 0.78,
		18: 0.79,
		20: 0.88,
		22: 0.86,
		24: 0.87,
		26: 0.93,
		28: 0.86,
		30: 0.95,
		32: 0.94,
		34: 0.94,
		36: 0.967,

		38: 0.799,
		40: 0.199,
		42: 0.659,
		44: 0.789,
		46: 0.909,
		48: 0.989,
	}

	return pre_x[n_features]

import numpy as np
from scipy.stats import uniform, expon
import random
from sklearn.datasets import make_regression
from scipy.stats import zscore
from scipy.special import expit

def reg_true_prob(n_samples, n_features, seed=0):
	# Set random seed
	np.random.seed(seed)
	random.seed(seed)

	X, y = make_regression(n_features=n_features, n_samples=n_samples)
	y_std = zscore(y)
	true_prob = expit(y_std)
	y = np.random.binomial(n=1, p=true_prob)
	return X, y, true_prob

def c_ng_p_chat(n_samples, 
				n_features, 
				class1_min=0, 
				class1_max=1, 
				class2_scale=1, 
				seed=0):
    """
    Generate synthetic data with true probabilities for binary classification using non-Gaussian distributions.

    Parameters:
    - n_samples: int, total number of samples.
    - n_features: int, number of features.
    - class1_min, class1_max: float, range for the uniform distribution of the first class.
    - class2_scale: float, scale parameter for the exponential distribution of the second class.
    - seed: int, random seed for reproducibility.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    - true_prob: np.ndarray, true probabilities for the labels.
    """
    # Number of samples per class
    n_samples_per_class = int(n_samples / 2)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Sample data from the uniform distribution for class 1
    x1 = np.random.uniform(class1_min, class1_max, (n_samples_per_class, n_features))
    
    # Sample data from the exponential distribution for class 2
    x2 = np.random.exponential(class2_scale, (n_samples_per_class, n_features))
    
    # Concatenate the data from both classes
    X = np.concatenate([x1, x2])
    
    # Calculate the true probabilities for class 2
    prob_class2 = np.prod(expon.pdf(X, scale=class2_scale), axis=1)
    prob_class1 = np.prod(uniform.pdf(X, loc=class1_min, scale=class1_max-class1_min), axis=1)
    true_prob = prob_class2 * 0.5 / (0.5 * prob_class1 + 0.5 * prob_class2)
    
    # Create labels for the data
    y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, true_prob


from scipy.stats import multivariate_normal, multivariate_t

def c_t_p_chat(n_samples, 
                n_features, 
                class1_mean_min=0, 
                class1_mean_max=1, 
                class1_cov_min=1, 
                class1_cov_max=2, 
                class2_mean_min=0, 
                class2_mean_max=1, 
                class2_cov_min=1, 
                class2_cov_max=2, 
                df=3,  # degrees of freedom for t-distribution
                seed=0):
    """
    Generate synthetic data with true probabilities for binary classification using multivariate t-distributions.

    Parameters:
    - n_samples: int, total number of samples.
    - n_features: int, number of features.
    - class1_mean_min, class1_mean_max: float, range for means of the first class.
    - class1_cov_min, class1_cov_max: float, range for covariances of the first class.
    - class2_mean_min, class2_mean_max: float, range for means of the second class.
    - class2_cov_min, class2_cov_max: float, range for covariances of the second class.
    - df: int, degrees of freedom for the t-distribution.
    - seed: int, random seed for reproducibility.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    - true_prob: np.ndarray, true probabilities for the labels.
    """
    # Number of samples per class
    n_samples_per_class = int(n_samples / 2)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate random mean and covariance for class 1
    mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features)
    cov1 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov1, np.random.uniform(class1_cov_min, class1_cov_max, n_features))
    
    # Generate random mean and covariance for class 2
    mean2 = np.random.uniform(class2_mean_min, class2_mean_max, n_features)
    cov2 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov2, np.random.uniform(class2_cov_min, class2_cov_max, n_features))
    
    # Sample data from the multivariate t-distributions
    x1 = multivariate_t.rvs(mean1, cov1, df, n_samples_per_class)
    x2 = multivariate_t.rvs(mean2, cov2, df, n_samples_per_class)
    
    # Concatenate the data from both classes
    X = np.concatenate([x1, x2])
    
    # Calculate the true probabilities for class 2
    prob_class2 = multivariate_t.pdf(X, mean2, cov2, df)
    prob_class1 = multivariate_t.pdf(X, mean1, cov1, df)
    true_prob = prob_class2 * 0.5 / (0.5 * prob_class1 + 0.5 * prob_class2)
    
    # Create labels for the data
    y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, true_prob


def c_g_p_chat(n_samples, 
				n_features, 
				class1_mean_min=0, 
				class1_mean_max=1, 
				class1_cov_min=1, 
				class1_cov_max=2, 
				class2_mean_min=0, 
				class2_mean_max=1, 
				class2_cov_min=1, 
				class2_cov_max=2, 
				seed=0):
    """
    Generate synthetic data with true probabilities for binary classification using Gaussian distributions.

    Parameters:
    - n_samples: int, total number of samples.
    - n_features: int, number of features.
    - class1_mean_min, class1_mean_max: float, range for means of the first class.
    - class1_cov_min, class1_cov_max: float, range for covariances of the first class.
    - class2_mean_min, class2_mean_max: float, range for means of the second class.
    - class2_cov_min, class2_cov_max: float, range for covariances of the second class.
    - seed: int, random seed for reproducibility.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    - true_prob: np.ndarray, true probabilities for the labels.
    """
    # Number of samples per class
    n_samples_per_class = int(n_samples / 2)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate random mean and covariance for class 1
    mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features)
    cov1 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov1, np.random.uniform(class1_cov_min, class1_cov_max, n_features))
    
    # Generate random mean and covariance for class 2
    mean2 = np.random.uniform(class2_mean_min, class2_mean_max, n_features)
    cov2 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov2, np.random.uniform(class2_cov_min, class2_cov_max, n_features))
    
    # Sample data from the multivariate normal distributions
    x1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)
    x2 = np.random.multivariate_normal(mean2, cov2, n_samples_per_class)
    
    # Concatenate the data from both classes
    X = np.concatenate([x1, x2])
    
    # Calculate the true probabilities for class 2
    prob_class2 = multivariate_normal.pdf(X, mean2, cov2)
    prob_class1 = multivariate_normal.pdf(X, mean1, cov1)
    true_prob = prob_class2 * 0.5 / (0.5 * prob_class1 + 0.5 * prob_class2)
    
    # Create labels for the data
    y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, true_prob

import numpy as np
import random
from scipy.stats import multivariate_normal

def c_g_p_chat_mixed_exp_gaussian(n_samples, 
                                  n_features, 
                                  class1_mean_min=0, 
                                  class1_mean_max=1, 
                                  class1_cov_min=1, 
                                  class1_cov_max=2, 
                                  class2_scale_min=0, 
                                  class2_scale_max=2, 
                                  seed=0):
    """
    Generate synthetic data with true probabilities for binary classification using multivariate exponential and normal distributions.

    Parameters:
    - n_samples: int, total number of samples.
    - n_features: int, number of features.
    - class1_mean_min, class1_mean_max: float, range for means of the first class.
    - class1_cov_min, class1_cov_max: float, range for covariances of the first class.
    - class2_scale_min, class2_scale_max: float, range for scales of the second class (exponential).
    - seed: int, random seed for reproducibility.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    - true_prob: np.ndarray, true probabilities for the labels.
    """
    # Number of samples per class
    n_samples_per_class = int(n_samples / 2)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate random mean and covariance for class 1 (multivariate normal)
    mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features)
    cov1 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov1, np.random.uniform(class1_cov_min, class1_cov_max, n_features))
    
    # Generate data for class 1
    x1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)
    
    # Generate scales for class 2 (exponential distribution)
    scales2 = np.random.uniform(class2_scale_min, class2_scale_max, n_features)
    
    # Generate data for class 2
    x2 = np.array([np.random.exponential(scale, n_samples_per_class) for scale in scales2]).T
    
    # Concatenate the data from both classes
    X = np.concatenate([x1, x2])
    
    # Calculate the true probabilities for class 2
    prob_class2 = np.prod([1/scale * np.exp(-X[:, i] / scale) for i, scale in enumerate(scales2)], axis=0)
    prob_class1 = multivariate_normal.pdf(X, mean1, cov1)
    true_prob = prob_class2 * 0.5 / (0.5 * prob_class1 + 0.5 * prob_class2)
    
    # Create labels for the data
    y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, true_prob

def c_gu_p_chat_mixed(n_samples, 
                     n_features, 
                     class1_mean_min=0, 
                     class1_mean_max=1, 
                     class1_cov_min=1, 
                     class1_cov_max=2, 
                     class2_min=0, 
                     class2_max=1, 
                     seed=0):
    """
    Generate synthetic data with true probabilities for binary classification using multivariate normal and uniform distributions.

    Parameters:
    - n_samples: int, total number of samples.
    - n_features: int, number of features.
    - class1_mean_min, class1_mean_max: float, range for means of the first class.
    - class1_cov_min, class1_cov_max: float, range for covariances of the first class.
    - class2_min, class2_max: float, range for values of the second class.
    - seed: int, random seed for reproducibility.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    - true_prob: np.ndarray, true probabilities for the labels.
    """
    # Number of samples per class
    n_samples_per_class = int(n_samples / 2)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate random mean and covariance for class 1 (multivariate normal)
    mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features)
    cov1 = np.zeros((n_features, n_features))
    np.fill_diagonal(cov1, np.random.uniform(class1_cov_min, class1_cov_max, n_features))
    
    # Generate data for class 1
    x1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)
    
    # Generate data for class 2 (multivariate uniform)
    x2 = np.random.uniform(class2_min, class2_max, (n_samples_per_class, n_features))
    
    # Concatenate the data from both classes
    X = np.concatenate([x1, x2])
    
    # Calculate the true probabilities for class 2
    prob_class2 = np.prod(1 / (class2_max - class2_min), axis=0)
    prob_class1 = multivariate_normal.pdf(X, mean1, cov1)
    true_prob = prob_class2 * 0.5 / (0.5 * prob_class1 + 0.5 * prob_class2)
    
    # Create labels for the data
    y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    
    return X, y, true_prob



def make_classification_gaussian_with_true_prob(n_samples, 
						n_features, 
						class1_mean_min=0, 
						class1_mean_max=1, 
						class1_cov_min=1, 
						class1_cov_max=2, 
						class2_mean_min=0, 
						class2_mean_max=1, 
						class2_cov_min=1, 
						class2_cov_max=2, 
						seed=0,
						): #0.76 only for #features exp
	n_samples = int(n_samples / 2)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	random.seed(seed)

	mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features) #[0, 2, 3, -1, 9]
	cov1 = np.zeros((n_features,n_features))
	np.fill_diagonal(cov1, np.random.uniform(class1_cov_min,class1_cov_max,n_features))
	# print("mean1", mean1)
	# print("cov1", cov1)

	mean2 = np.random.uniform(class2_mean_min, class2_mean_max,n_features) # [-1, 3, 0, 2, 3]
	cov2 = np.zeros((n_features,n_features))
	np.fill_diagonal(cov2, np.random.uniform(class2_cov_min,class2_cov_max,n_features))
	# print("mean2", mean2)
	# print("cov2", cov2)

	x1 = np.random.multivariate_normal(mean1, cov1, n_samples)
	x2 = np.random.multivariate_normal(mean2, cov2, n_samples)

	X = np.concatenate([x1, x2])
	true_prob = multivariate_normal.pdf(X, mean2, cov2) * 0.5 / (0.5 * multivariate_normal.pdf(X, mean1, cov1) + 0.5 * multivariate_normal.pdf(X, mean2, cov2))
	y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
	
	overlap_delta = mean2.mean() - mean1.mean() # if cov1 and cov2 are the same

	return X, y, true_prob , overlap_delta


from scipy.linalg import sqrtm

def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    """
    Compute the Bhattacharyya distance between two multivariate Gaussian distributions.
    
    :param mean1: Mean vector of the first distribution.
    :param cov1: Covariance matrix of the first distribution.
    :param mean2: Mean vector of the second distribution.
    :param cov2: Covariance matrix of the second distribution.
    :return: Bhattacharyya distance.
    """
    mean_diff = mean2 - mean1
    
    # Calculate the average of the covariance matrices
    cov_mean = (cov1 + cov2) / 2
    
    # Calculate the Bhattacharyya distance
    cov_inv_mean = np.linalg.inv(cov_mean)
    mean_term = 0.125 * np.dot(np.dot(mean_diff.T, cov_inv_mean), mean_diff)
    cov_term = 0.5 * np.log(np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    
    return mean_term + cov_term


def make_classification_gaussian_with_true_prob_overlap(n_samples, 
						n_features, 
						class1_mean_min=0, 
						class1_mean_max=1, 
						class1_cov_min=1, 
						class1_cov_max=2, 
						delta = 0.25,
						seed=0,
						):
	n_samples = int(n_samples / 2)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	random.seed(seed)

	mean1 = np.random.uniform(class1_mean_min, class1_mean_max, n_features) #[0, 2, 3, -1, 9]
	cov1 = np.zeros((n_features,n_features))
	np.fill_diagonal(cov1, np.random.uniform(class1_cov_min,class1_cov_max,n_features))
	# print("mean1", mean1)
	# print("cov1", cov1)

	mean2 = mean1 + delta 
	cov2 = cov1
	# print("mean2", mean2)
	# print("cov2", cov2)

	np.random.seed(seed)
	random.seed(seed)

	x1 = np.random.multivariate_normal(mean1, cov1, n_samples)

	# np.random.seed(seed)
	# random.seed(seed)

	x2 = np.random.multivariate_normal(mean2, cov2, n_samples)

	X = np.concatenate([x1, x2])
	true_prob = multivariate_normal.pdf(X, mean2, cov2) * 0.5 / (0.5 * multivariate_normal.pdf(X, mean1, cov1) + 0.5 * multivariate_normal.pdf(X, mean2, cov2))
	y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
	
	# overlap_delta = mean2.mean() - mean1.mean() # if cov1 and cov2 are the same
	# print("overlap_delta", overlap_delta)

	overlap_delta = bhattacharyya_distance(mean1, cov1, mean2, cov2)

	return X, y, true_prob , overlap_delta


def make_classification_gaussian_with_true_prob_control_RF_acc(n_samples, 
						n_features, 
						class1_mean_min=0, 
						class1_mean_max=1, 
						class1_cov_min=1, 
						class1_cov_max=2, 
						class2_mean_min=0, 
						class2_mean_max=1, 
						class2_cov_min=1, 
						class2_cov_max=2, 
						seed=0,
						bais_accuracy= 0): #0.76 only for #features exp
	n_samples = int(n_samples / 2)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	random.seed(seed)

	x_list = np.arange(0, 1, 0.001)
	x_list = np.round(x_list, decimals=3)

	for x in x_list:
		xx = x
		if x == 0 and bais_accuracy != 0:
			xx = get_pre_x(n_features)
		np.random.seed(int(xx* 100)) # change to xx later
		if n_features > 36 and bais_accuracy != 0:
			xx = 1

		mean1 = np.random.uniform(class1_mean_min + xx, class1_mean_max + xx, n_features) #[0, 2, 3, -1, 9]
		cov1 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov1, np.random.uniform(class1_cov_min,class1_cov_max,n_features))

		mean2 = np.random.uniform(class2_mean_min - xx, class2_mean_max - xx,n_features) # [-1, 3, 0, 2, 3]
		cov2 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov2, np.random.uniform(class2_cov_min,class2_cov_max,n_features))

		x1 = np.random.multivariate_normal(mean1, cov1, n_samples)
		x2 = np.random.multivariate_normal(mean2, cov2, n_samples)

		X = np.concatenate([x1, x2])
		true_prob = multivariate_normal.pdf(X, mean2, cov2) * 0.5 / (0.5 * multivariate_normal.pdf(X, mean1, cov1) + 0.5 * multivariate_normal.pdf(X, mean2, cov2))
		y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])


		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
		clf = RandomForestClassifier(n_estimators=100, random_state=seed)  

		# print(">>> f", n_features, " x ", xx)
  
		clf.fit(x_train, y_train)
		accuracy = clf.score(x_test, y_test)
		print(f"{n_features}: {xx} ACC {accuracy}")

		if bais_accuracy == 0:
			break
		if accuracy < bais_accuracy and accuracy > bais_accuracy - 0.05:
			# print(f"{n_features}: {x},")
			# if xx != get_pre_x(n_features):
			# 	print("pre_x did not work")
			break

	return X, y, true_prob

def make_classification_mixture_gaussian_with_true_prob(n_samples, 
						n_features,
						n_clusters, 
						same_cov = True,
						seed=0,
						bais_accuracy= 0): #0.76 only for #features exp
	n_samples = int(n_samples / 2 / n_clusters)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	
	means1 = []
	covariances1 = []
	weights1 = []

	means2 = []
	covariances2 = []
	weights2 = []

	for i in range(n_clusters):
		mean_min_max = np.random.randint(1, 20, size=2)
		cov_min_max = np.random.randint(1, 5, size=2)
		means1.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
		cov1 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov1, np.random.uniform(cov_min_max.min(), cov_min_max.max(),n_features))
		covariances1.append(cov1)
		weights1.append(1 / n_clusters)  # Equal weights1 for all components

		mean_min_max = np.random.randint(1, 20, size=2)
		cov2_min_max = np.random.randint(1, 5, size=2)
		if same_cov:
			cov2_min_max = cov_min_max
		means2.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
		cov2 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov2, np.random.uniform(cov2_min_max.min(), cov2_min_max.max(),n_features))
		covariances2.append(cov2)
		weights2.append(1 / n_clusters)  # Equal weights for all components

	data1 = []
	for mean, covariance in zip(means1, covariances1):
		data1.append(np.random.multivariate_normal(mean, covariance, n_samples))

	data2 = []
	for mean, covariance in zip(means2, covariances2):
		data2.append(np.random.multivariate_normal(mean, covariance, n_samples))

	x1 = np.concatenate(data1, axis=0)
	x2 = np.concatenate(data2, axis=0)
	X = np.concatenate([x1, x2])

	# Calculate the PDF of the mixture Gaussian distributions
	mixture_pdf1 = np.zeros(len(X))
	for mean, covariance, weight in zip(means1, covariances1, weights1):
		mixture_pdf1 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

	mixture_pdf2 = np.zeros(len(X))
	for mean, covariance, weight in zip(means2, covariances2, weights2):
		mixture_pdf2 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

	# Calculate the probabilities of each sample belonging to class 1
	prob_class1 = mixture_pdf1 * 0.5  # Class 1 prior probability is 0.5
	prob_class2 = mixture_pdf2 * 0.5  # Class 2 prior probability is 0.5

	# Split true probabilities into two parts, one for each class
	true_prob = prob_class2 / (prob_class1 + prob_class2)

	y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

	return X, y, true_prob

# def make_classification_mixture_gaussian_with_true_prob(n_samples, 
# 						n_features,
# 						n_clusters, 
# 						same_cov = True,
# 						seed=0,
# 						bais_accuracy= 0): #0.76 only for #features exp
# 	n_samples = int(n_samples / 2)
# 	# Synthetic data with n_features dimentions and n_classes classes

# 	np.random.seed(seed)
	
# 	means1 = []
# 	covariances1 = []
# 	weights1 = []

# 	means2 = []
# 	covariances2 = []
# 	weights2 = []

# 	for i in range(n_clusters):
# 		mean_min_max = np.random.randint(1, 20, size=2)
# 		cov_min_max = np.random.randint(1, 5, size=2)
# 		means1.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
# 		cov1 = np.zeros((n_features,n_features))
# 		np.fill_diagonal(cov1, np.random.uniform(cov_min_max.min(), cov_min_max.max(),n_features))
# 		covariances1.append(cov1)
# 		weights1.append(1 / n_clusters)  # Equal weights1 for all components

# 		mean_min_max = np.random.randint(1, 20, size=2)
# 		cov2_min_max = np.random.randint(1, 5, size=2)
# 		if same_cov:
# 			cov2_min_max = cov_min_max
# 		means2.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
# 		cov2 = np.zeros((n_features,n_features))
# 		np.fill_diagonal(cov2, np.random.uniform(cov2_min_max.min(), cov2_min_max.max(),n_features))
# 		covariances2.append(cov2)
# 		weights2.append(1 / n_clusters)  # Equal weights for all components

# 	data1 = []
# 	for mean, covariance in zip(means1, covariances1):
# 		data1.append(np.random.multivariate_normal(mean, covariance, n_samples))

# 	data2 = []
# 	for mean, covariance in zip(means2, covariances2):
# 		data2.append(np.random.multivariate_normal(mean, covariance, n_samples))

# 	x1 = np.concatenate(data1, axis=0)
# 	x2 = np.concatenate(data2, axis=0)
# 	X = np.concatenate([x1, x2])

# 	# Calculate the PDF of the mixture Gaussian distributions
# 	mixture_pdf1 = np.zeros(len(X))
# 	for mean, covariance, weight in zip(means1, covariances1, weights1):
# 		mixture_pdf1 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

# 	mixture_pdf2 = np.zeros(len(X))
# 	for mean, covariance, weight in zip(means2, covariances2, weights2):
# 		mixture_pdf2 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

# 	# Calculate the probabilities of each sample belonging to class 1
# 	prob_class1 = mixture_pdf1 * 0.5  # Class 1 prior probability is 0.5
# 	prob_class2 = mixture_pdf2 * 0.5  # Class 2 prior probability is 0.5

# 	# Split true probabilities into two parts, one for each class
# 	true_prob = prob_class2 / (prob_class1 + prob_class2)

# 	y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

# 	return X, y, true_prob

from sklearn.datasets import make_regression

def make_classification_with_true_prob2(n_samples, n_features, n_classes=2, seed=0):
	X, tp = make_regression(n_samples, n_features, tail_strength=0) # make regression data
	y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold
	return X, y, tp

def make_classification_with_true_prob3(n_samples, w=2, noise_mu=0, noise_sigma=0.1, seed=0):
	# y = x.w + noise
	n = np.random.normal(noise_mu, noise_sigma, n_samples)
	x = np.random.uniform(-1,1,n_samples)
	tp = x * w + n

	y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold
	return x.reshape(-1, 1), y, x

def make_classification_with_true_prob_logestic(n_samples, n_features, mean_true_prob=0.8, std_true_prob= 0.2, seed=0):
	np.random.seed(seed)

	true_prob = np.random.normal(loc=mean_true_prob, scale=std_true_prob, size=n_samples)
	true_prob = np.where(true_prob<0, 0, true_prob) # clip negetive values
	true_prob = np.where(true_prob>1, 1, true_prob) # clip greater than 1

	logit = np.log(true_prob / (1-true_prob))

	mean1 = np.random.uniform(-1,1,n_features) #[0, 2, 3, -1, 9]
	cov1 = np.zeros((n_features,n_features))
	np.fill_diagonal(cov1, np.random.uniform(0,1,n_features))
	X = np.random.multivariate_normal(mean1, cov1, n_samples)

	beta_cof = np.random.uniform(-1,1,n_features)

	alpha = logit - np.mean(beta_cof * X, axis=1)

	y = []
	for tp in true_prob:
		y.append(np.random.binomial(1, tp , 1))
	# logit = alpha + beta_cof * X
	# true_prob = 1/(1 + np.exp(-logit))

	return X, y, true_prob 

def make_classification_with_true_prob_3(n_samples, n_features, seed=0):
	# Set random seed for reproducibility
	np.random.seed(seed)
	random.seed(seed)

	# Generate synthetic dataset
	# X, Y = make_classification(n_samples=n_samples, n_features=n_features, random_state=seed)
	X, Y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=2, n_clusters_per_class=1, random_state=42)

	# Generate true probabilities for each instance

	weights = np.random.rand(X.shape[1])
	bias = np.random.rand()
	linear_combination = np.dot(X, weights) + bias
	# Normalize to [0, 1] using sigmoid function
	true_probabilities = 1 / (1 + np.exp(-linear_combination))


	return X, Y, true_probabilities 



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntheticDataGenerator:
	def __init__(self, num_features=10, num_classes=2, hidden_layers=[32, 16], seed=42):
		"""
		Initialize the synthetic data generator.

		Args:
		num_features (int): Number of input features (X).
		num_classes (int): Number of output classes (K). Use 2 for binary classification.
		hidden_layers (list): List specifying the number of neurons in each hidden layer.
		seed (int): Random seed for reproducibility.
		"""
		self.num_features = num_features
		self.num_classes = num_classes
		self.seed = seed

		torch.manual_seed(seed)
		np.random.seed(seed)

		# Define the neural network model
		layers = []
		input_dim = num_features

		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(input_dim, hidden_dim))
			layers.append(nn.ReLU())
			input_dim = hidden_dim

		# Output layer
		layers.append(nn.Linear(input_dim, num_classes))

		self.model = nn.Sequential(*layers)

		# Initialize weights randomly
		for layer in self.model:
			if isinstance(layer, nn.Linear):
				nn.init.kaiming_normal_(layer.weight)

	def generate_data(self, num_samples=1000, temperature=1.0, mask_ratio=0.0, x_grid=False):
		"""
		Generate synthetic data.

		Args:
		num_samples (int): Number of samples to generate.
		temperature (float): Softmax temperature to control classification difficulty.
		mask_ratio (float): Ratio of features to mask (turn into noise).

		Returns:
		X (numpy.ndarray): Generated feature matrix (num_samples, num_features).
		Y (numpy.ndarray): Generated labels (num_samples,).
		"""
		# Generate random feature matrix X
		X = np.random.randn(num_samples, self.num_features).astype(np.float32)
		if x_grid:
			X_g = make_grid_data(100, X).astype(np.float32)

		# Mask certain features (turn into noise)
		if mask_ratio > 0:
			num_masked = int(self.num_features * mask_ratio)
			mask_indices = np.random.choice(self.num_features, num_masked, replace=False)
			X[:, mask_indices] = np.random.permutation(X[:, mask_indices])  # Shuffle to add noise
			if x_grid:
				X_g[:, mask_indices] = np.random.permutation(X_g[:, mask_indices])  # Shuffle to add noise

		X_tensor = torch.tensor(X)
		if x_grid:
			X_g_tensor = torch.tensor(X_g)

		# Forward pass through the neural network to get logits
		logits = self.model(X_tensor)
		if x_grid:
			logits_g = self.model(X_g_tensor)

		# Normalize outputs for each class between 0 and 1 before softmax
		logits_min = logits.min(dim=0, keepdim=True).values
		logits_max = logits.max(dim=0, keepdim=True).values
		normalized_logits = (logits - logits_min) / (logits_max - logits_min + 1e-8)  # Adding epsilon for numerical stability

		# Normalize outputs for each class between 0 and 1 before softmax
		if x_grid:
			logits_g_min = logits_g.min(dim=0, keepdim=True).values
			logits_g_max = logits_g.max(dim=0, keepdim=True).values
			normalized_logits_g = (logits_g - logits_g_min) / (logits_g_max - logits_g_min + 1e-8)  # Adding epsilon for numerical stability


		# Apply softmax with temperature scaling
		P = F.softmax(normalized_logits / temperature, dim=1).detach().numpy()
		if x_grid:
			P_g = F.softmax(normalized_logits_g / temperature, dim=1).detach().numpy()

		# Sample Y from categorical distribution
		if self.num_classes == 2:
			Y = np.random.binomial(1, P[:, 1])  # Take the probability of class 1
		else:
			Y = np.array([np.random.choice(self.num_classes, p=p_row) for p_row in P])

		if x_grid:
			return X, Y, P, X_g, P_g

		return X, Y, P