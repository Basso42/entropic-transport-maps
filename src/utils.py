import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def chamfer_distance(point_cloud1, point_cloud2):
    pairwise_distances = cdist(point_cloud1, point_cloud2, metric='euclidean')
    total_distance = np.sum(np.min(pairwise_distances, axis=1))
    return total_distance / len(point_cloud1)

def plot_2d(X, Y, title="2D Plot", labels=('Source', 'Target', 'Transported')):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], label=labels[0], alpha=0.5)
    plt.scatter(Y[:, 0], Y[:, 1], label=labels[1], alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d(X, Y, Z=None, title="3D Plot", labels=('Source', 'Target', 'Transported')):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], label=labels[0], alpha=0.6)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], label=labels[1], alpha=0.6)
    if Z is not None:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], label=labels[2], alpha=0.5)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
