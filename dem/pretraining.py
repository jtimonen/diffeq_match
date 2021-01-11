from sklearn.manifold import Isomap


def isomap(X, **isomap_kwargs):
    """Create and fit an Isomap object."""
    if isomap_kwargs is None:
        isomap_kwargs = dict(n_components=2)
    embedding = Isomap(**isomap_kwargs)
    embedding.fit(X)
    return embedding


def gdist(X, n_neighbors=5):
    """Compute geodesic distances between all rows of X.

    :param n_neighbors: argument for sklearn.manifold.Isomap
    :param X: Sample data, shape = (n_samples, n_features), in the form of a
        numpy array, sparse graph, precomputed tree, or NearestNeighbors object.
    :return: a distance matrix with shape = (n_samples, n_samples)
    """
    isomap_kwargs = dict(n_neighbors=n_neighbors)
    embedding = isomap(X, **isomap_kwargs)
    return embedding.dist_matrix_
