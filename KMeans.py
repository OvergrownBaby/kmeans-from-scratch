import numpy as np
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, n_clusters: int = 8, init: str = 'forgy', max_iter: int = 300, tol = 1e-4, vectorize: bool = True, elkan: bool = False):
        self.k = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        self.elkan = elkan
        self.vectorize = vectorize
        self.tol = tol
    
    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_clusters = X.shape
        self.centroids = np.zeros((self.k, n_clusters))
        match self.init:
            case 'forgy':
                random_indices = np.random.choice(n_samples, size=self.k, replace=False)
                self.centroids = X[random_indices, :]
            case 'k-means++':
                self.centroids[0] = X[np.random.choice(n_samples)]
                distances = np.full(n_samples, np.inf)
                for i in range(1, self.k):
                    # distance between each point and the latest added centroid
                    dist_sq = cdist(self.centroids[i - 1:i], X, metric='sqeuclidean').flatten()
                    # dist_sq = np.linalg.norm(self.centroids[i - 1] - X, axis=1) ** 2
                    distances = np.minimum(distances, dist_sq) # update the least distance
                    probabilities = distances / distances.sum() # select the next centroid
                    # select the next centroid
                    
                    # ### slow but easy to understand
                    # cumulative_probabilities = np.cumsum(probabilities)
                    # r = np.random.rand()
                    # for j, p in enumerate(cumulative_probabilities):
                    #     if r < p:
                    #         self.centroids[i] = X[j]
                    #         break

                    ### faster method
                    new_centroid_index = np.random.choice(n_samples, p=probabilities)
                    self.centroids[i] = X[new_centroid_index]
        if self.elkan:
            #under contruction
            pass
        else:
            for _ in range(self.max_iter):
                ### with vectorization
                if self.vectorize:
                    distances = cdist(X, self.centroids)
                    # distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
                    nearest_cluster_ids = np.argmin(distances, axis=1)
                    new_centroids = np.array([X[nearest_cluster_ids==k].mean(axis=0) if np.any(nearest_cluster_ids==k) else X[np.random.choice(n_samples)] for k in range(self.k)])
                    if np.allclose(self.centroids, new_centroids):
                    # if np.array_equal(self.centroids, new_centroids):
                        break
                    self.centroids = new_centroids

                ### no vectorization
                if not self.vectorize:
                    # assign each point to the nearest centroid
                    clusters = [[] for _ in range(self.k)]
                    clusters_changed = [False] * self.k
                    for i in range(n_samples):
                        distances = cdist(X, self.centroids)
                        # distances = np.linalg.norm(self.centroids - X[i], axis=1)
                        clusters[np.argmin(distances)].append(i)
                        clusters_changed[np.argmin(distances)] = True


                    new_centroids = np.zeros_like(self.centroids)

                    # update the centroids
                    for i in range(self.k):
                        if clusters[i] and clusters_changed[i]:
                            new_centroids[i] = np.mean(np.matrix(X[clusters[i]]), axis=0)
                        else:
                            new_centroids[i] = self.centroids[i]

                    # check for convergence
                    if np.array_equal(self.centroids, new_centroids, atol=self.tol):
                        break
                    else:
                        self.centroids = new_centroids
            

    
    def predict(self, X):
        X = np.matrix(X)
        y_pred = []
        n_samples = X.shape[0]
        for i in range(n_samples):
            distances = np.linalg.norm(self.centroids - X[i], axis=1)
            
            if np.argmin(distances) is None:
                print(distances)
                raise ValueError("np.argmin(distances) returned None.")
            else:
                y_pred.append(np.argmin(distances))
        return np.array(y_pred)