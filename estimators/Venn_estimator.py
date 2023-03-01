import numpy as np
import heapq


class Venn_calib():

    def fit(self, X, y):
        
        preds = np.argmax(X, axis=1)
        classes = np.unique(preds)
        n_samples = len(preds)
        classes_map = {}
        max_lower, max_upper = [], []

        # create a map
        # keys are classes
        # values ore lists of (prediction,label) pairs
        for c in classes:
            classes_map[c] = []

        for i in range(n_samples):
            classes_map[preds[i]].append((preds[i], y[i]))

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

        self.l = np.asarray(max_lower)
        self.u = np.asarray(max_upper)

        return self
    
    
    def predict(self, X):
        
        preds = np.argmax(X, axis=1)
        preds = np.array(preds, dtype=float)

        classes = np.unique(preds)
        classes = np.array(classes, dtype=int)

        probs = (self.l + self.u) / 2

        for c in classes:
            np.place(preds, preds==c, probs[c])
        
        return preds