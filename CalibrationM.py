import numpy as np
from math import isnan
from sklearn.metrics import brier_score_loss

def classwise_ECE(probs, y_true, bins=10, equal_bin_size=True, full_ece=False):
    
    class_ece = []

    for class_index in range(len(np.unique(y_true))):
        ece = 0
        class_prob = probs[:,class_index] # select probs form i th class
        correctness_map = np.where(class_index==y_true, 1, 0) # determine which predictions are correct

        if equal_bin_size == False:
            bin_size = 1/bins
            for bin in range(bins):
                bin_indexes = np.where((class_prob > bin * bin_size) & (class_prob <= (bin+1) * bin_size))[0]
                if len(bin_indexes) > 0:
                    bin_conf = class_prob[bin_indexes].mean()
                    bin_class_friq = correctness_map[bin_indexes].sum() / len(bin_indexes)
                    dif = abs(bin_conf - bin_class_friq)
                    ece += dif * len(bin_indexes) / len(probs)
                class_ece.append(ece)  # difference times the number of instances in the bin divided by the total number of instances in the dataset
        else: # equal_bin_size
            sorted_index = np.argsort(class_prob, kind='stable') # sort probs
            class_prob = class_prob[sorted_index]
            correctness_map = correctness_map[sorted_index]

            bin_size = int(len(probs) / bins)
            for bin in range(bins):
                bin_conf = class_prob[bin*bin_size:(bin+1)*bin_size].mean()
                bin_class_friq = correctness_map[bin*bin_size:(bin+1)*bin_size].sum() / bin_size
                dif = abs(bin_conf - bin_class_friq)
                ece += dif * bin_size / len(probs)
            class_ece.append(ece)

    if full_ece:
        return sum(class_ece)
    else:
        return class_ece 

def confidance_ECE(probs, y_true, bins=10, equal_bin_size=True):
    prob_max = np.max(probs, axis=1) # find the most probabil class
    correctness_map = np.where(np.argmax(probs, axis=1)==y_true, 1, 0) # determine which predictions are correct
    ece = 0

    if equal_bin_size == False:
        bin_size = 1/bins
        for bin in range(bins):
            bin_indexes = np.where((prob_max > bin * bin_size) & (prob_max <= (bin+1) * bin_size))[0]
            if len(bin_indexes) > 0:
                bin_conf = prob_max[bin_indexes].mean()
                bin_acc = correctness_map[bin_indexes].sum() / len(bin_indexes)
                dif = abs(bin_conf - bin_acc)
                ece += dif * len(bin_indexes) / len(probs)  # difference times the number of instances in the bin divided by the total number of instances in the dataset
    else: # equal_bin_size
        sorted_index = np.argsort(prob_max, kind='stable') # sort probs
        prob_max = prob_max[sorted_index]
        correctness_map = correctness_map[sorted_index]

        bin_size = int(len(y_true) / bins)
        for bin in range(bins):
            bin_conf = prob_max[bin*bin_size:(bin+1)*bin_size].mean()
            bin_acc = correctness_map[bin*bin_size:(bin+1)*bin_size].sum() / bin_size
            # print(f">>> bin_conf {bin_conf} bin_acc {bin_acc} ")
            dif = abs(bin_conf - bin_acc)
            if isnan(dif):
                break
            ece += dif * bin_size / len(probs)

    return ece 

def Reliability(probs, y_true, bins=10):
    prob_max = np.max(probs, axis=1) # find the most probabil class
    correctness_map = np.where(np.argmax(probs, axis=1)==y_true, 1, 0) # determine which predictions are correct
    ece = 0

    bin_size = 1/bins
    for bin in range(bins):
        bin_indexes = np.where((prob_max > bin * bin_size) & (prob_max <= (bin+1) * bin_size))[0]
        if len(bin_indexes) > 0:
            bin_conf = prob_max[bin_indexes].mean()
            bin_acc = correctness_map[bin_indexes].sum() / len(bin_indexes)
            dif = abs(bin_conf - bin_acc)
            ece += dif * len(bin_indexes) / len(probs)  # difference times the number of instances in the bin divided by the total number of instances in the dataset

    return ece 

def convert_prob_2D(prob1D):
    prob_second_class = np.ones(len(prob1D)) - prob1D
    prob2D = np.concatenate((prob_second_class.reshape(-1,1), prob1D.reshape(-1,1)), axis=1)
    return prob2D