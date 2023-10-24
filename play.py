from scipy.stats import ttest_ind, ttest_rel
import numpy as np

x = np.arange(30)
y = x + 0.1

statistic, p_value = ttest_ind(x, y)

print("p_value", p_value)