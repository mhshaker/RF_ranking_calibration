import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mu, sigma = [0, 2], [0.1, 0.5]
s1 = np.random.normal(mu[0], sigma[0], 1000)
s1 = np.sort(s1) # just for visula

s2 = np.random.normal(mu[1], sigma[1], 1000)
s2 = np.sort(s2) # just for visula

s1_pdf_dif = norm.pdf(s1, mu[0], sigma[0]) - norm.pdf(s1, mu[1], sigma[1])
s2_pdf_dif = norm.pdf(s2, mu[1], sigma[1]) - norm.pdf(s2, mu[0], sigma[0])

X = np.concatenate([s1, s2])
y = np.concatenate([np.zeros(len(s1)), np.ones(len(s2))])
t_p = np.concatenate([s1_pdf_dif, s2_pdf_dif])

print("x", t_p.shape)

# plt.plot(s1_pdf)
# plt.plot(s2_pdf)

# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
# plt.show()




# from sklearn.utils import check_random_state
# from sklearn.utils import shuffle as util_shuffle
# from sklearn.utils.random import sample_without_replacement
# from sklearn.datasets._samples_generator import _generate_hypercube


# def make_classification(
#     n_samples=100,
#     n_features=20,
#     *,
#     n_informative=2,
#     n_redundant=2,
#     n_repeated=0,
#     n_classes=2,
#     n_clusters_per_class=2,
#     weights=None,
#     flip_y=0.01,
#     class_sep=1.0,
#     hypercube=True,
#     shift=0.0,
#     scale=1.0,
#     shuffle=True,
#     random_state=None,
# ):
#     generator = check_random_state(random_state)

#     # Count features, clusters and samples
#     if n_informative + n_redundant + n_repeated > n_features:
#         raise ValueError(
#             "Number of informative, redundant and repeated "
#             "features must sum to less than the number of total"
#             " features"
#         )
#     # Use log2 to avoid overflow errors
#     if n_informative < np.log2(n_classes * n_clusters_per_class):
#         msg = "n_classes({}) * n_clusters_per_class({}) must be"
#         msg += " smaller or equal 2**n_informative({})={}"
#         raise ValueError(
#             msg.format(
#                 n_classes, n_clusters_per_class, n_informative, 2**n_informative
#             )
#         )

#     if weights is not None:
#         if len(weights) not in [n_classes, n_classes - 1]:
#             raise ValueError(
#                 "Weights specified but incompatible with number of classes."
#             )
#         if len(weights) == n_classes - 1:
#             if isinstance(weights, list):
#                 weights = weights + [1.0 - sum(weights)]
#             else:
#                 weights = np.resize(weights, n_classes)
#                 weights[-1] = 1.0 - sum(weights[:-1])
#     else:
#         weights = [1.0 / n_classes] * n_classes

#     n_useless = n_features - n_informative - n_redundant - n_repeated
#     n_clusters = n_classes * n_clusters_per_class

#     # Distribute samples among clusters by weight
#     n_samples_per_cluster = [
#         int(n_samples * weights[k % n_classes] / n_clusters_per_class)
#         for k in range(n_clusters)
#     ]

#     for i in range(n_samples - sum(n_samples_per_cluster)):
#         n_samples_per_cluster[i % n_clusters] += 1

#     # Initialize X and y
#     X = np.zeros((n_samples, n_features))
#     y = np.zeros(n_samples, dtype=int)

#     # Build the polytope whose vertices become cluster centroids
#     centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
#         float, copy=False
#     )
#     centroids *= 2 * class_sep
#     centroids -= class_sep
#     if not hypercube:
#         centroids *= generator.uniform(size=(n_clusters, 1))
#         centroids *= generator.uniform(size=(1, n_informative))

#     # Initially draw informative features from the standard normal
#     X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))

#     # Create each cluster; a variant of make_blobs
#     stop = 0
#     for k, centroid in enumerate(centroids):
#         start, stop = stop, stop + n_samples_per_cluster[k]
#         y[start:stop] = k % n_classes  # assign labels
#         X_k = X[start:stop, :n_informative]  # slice a view of the cluster

#         A = 2 * generator.uniform(size=(n_informative, n_informative)) - 1
#         X_k[...] = np.dot(X_k, A)  # introduce random covariance

#         X_k += centroid  # shift the cluster to a vertex

#     # Create redundant features
#     if n_redundant > 0:
#         B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
#         X[:, n_informative : n_informative + n_redundant] = np.dot(
#             X[:, :n_informative], B
#         )

#     # Repeat some features
#     if n_repeated > 0:
#         n = n_informative + n_redundant
#         indices = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(np.intp)
#         X[:, n : n + n_repeated] = X[:, indices]

#     # Fill useless features
#     if n_useless > 0:
#         X[:, -n_useless:] = generator.standard_normal(size=(n_samples, n_useless))

#     # Randomly replace labels
#     if flip_y >= 0.0:
#         flip_mask = generator.uniform(size=n_samples) < flip_y
#         y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

#     # Randomly shift and scale
#     if shift is None:
#         shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
#     X += shift

#     if scale is None:
#         scale = 1 + 100 * generator.uniform(size=n_features)
#     X *= scale

#     if shuffle:
#         # Randomly permute samples
#         X, y = util_shuffle(X, y, random_state=generator)

#         # Randomly permute features
#         indices = np.arange(n_features)
#         generator.shuffle(indices)
#         X[:, :] = X[:, indices]

#     return X, y