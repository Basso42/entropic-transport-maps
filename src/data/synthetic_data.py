from sklearn.datasets import make_blobs, make_swiss_roll


def generate_blobs(n_samples=1000, centers=2, cluster_std=1.0, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, y

def generate_swiss_roll(n_samples=1000, noise=0.0, random_state=42):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, t

