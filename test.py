from scipy import stats
x1 = [1,2,3,4,5]
x2 = [5,4,30,2,1]
tau, p_value = stats.kendalltau(x1, x2)
print("tau", tau)
