import numpy as np
import heapq


class Venn_calib():

    def Venn_predictor(preds, labels):

        classes = np.unique(preds)
        n_test_samples = len(preds)
        classes_map = {}
        max_lower, max_upper = [], []
        final_probs = []

        # create a map
        # keys are classes
        # values ore lists of (prediction,label) pairs
        for c in classes:
            classes_map[c] = []

        for i in range(n_test_samples):
            classes_map[preds[i]].append((preds[i], labels[i]))

        # iterate on pairs of each class and compute lower and upper for it
        for pairs in classes_map.values():

            lower = []
            upper = []
            labels_count = {}
            
            for c in classes:
                labels_count[c] = 0

            for pair in pairs:
                labels_count[pair[1]] += 1

            for c in classes:

                low = labels_count[c] / (len(pairs)+1)
                up = (labels_count[c]+1) / (len(pairs)+1)

                lower.append(low)
                upper.append(up)  

            heapq.heapify(lower)
            heapq.heapify(upper)  

            # get the maximum prob for each class
            max_lower.append(lower[-1])
            max_upper.append(upper[-1])

            # compute final prob using upper and lower
            final_prob = (lower[-1] + upper[-1]) / 2
            final_probs.append(final_prob)

        return np.asarray(final_probs)
