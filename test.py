import numpy as np
from scipy.stats import multivariate_normal

samples = 1000

mean1 = [0, 2, 3, -1, 9]
cov1 = [[.1, 0, 0, 0, 0], 
        [0, .5, 0, 0, 0],
        [0, 0, 0.8, 0, 0],
        [0, 0, 0, .1, 0],
        [0, 0, 0, 0, .3],
        ]

mean2 = [-1, 3, 0, 2, 3]
cov2 = [[.9, 0, 0, 0, 0], 
        [0, .1, 0, 0, 0],
        [0, 0, 0.3, 0, 0],
        [0, 0, 0, .1, 0],
        [0, 0, 0, 0, .7],
        ]


x1 = np.random.multivariate_normal(mean1, cov1, samples)
x2 = np.random.multivariate_normal(mean2, cov2, samples)

x1_pdf_dif = multivariate_normal.pdf(x1, mean1, cov1) - multivariate_normal.pdf(x1, mean2, cov2)
x2_pdf_dif = multivariate_normal.pdf(x2, mean2, cov2) - multivariate_normal.pdf(x2, mean1, cov1)

X = np.concatenate([x1, x2])
y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
t_p = np.concatenate([x1_pdf_dif, x2_pdf_dif])


print("X", X.shape)
print("y", y.shape)
print("t_p", t_p.shape)