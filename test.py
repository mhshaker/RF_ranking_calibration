from scipy import stats
x1 = [1,2,3,4,5]
x2 = [1,2,3,4,5]
tau, p_value = stats.kendalltau(x1, x2)
print("tau", tau)
