import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic


rng = np.random.default_rng()
windspeed = 8 * rng.random(5000)
print("windspeed", windspeed)
boatspeed = .3 * windspeed**.5 + .2 * rng.random(5000)
bin_means, bin_edges, binnumber = binned_statistic(windspeed, boatspeed, bins=100)
plt.figure()
plt.scatter(windspeed, boatspeed, label='raw data')
plt.scatter((bin_edges[:-1] + bin_edges[1:])/2, bin_means, label='binned statistic of data')
# plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='black', lw=5, label='binned statistic of data')
plt.legend()
plt.show()