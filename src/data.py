import numpy as np
import jax.numpy as jnp
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

def generate_blobs(n_samples=1000):
    X, _ = make_blobs(n_samples=n_samples, centers=[[0, 0]], cluster_std=1.0, random_state=42)
    Y, _ = make_blobs(n_samples=n_samples, centers=[[0, 10], [0, -10]], cluster_std=1.0, random_state=43)
    return jnp.array(X), jnp.array(Y)

def generate_moons(n_samples=1000):
    X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    Y, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    Y += np.array([3, 3]) #offset
    return jnp.array(X), jnp.array(Y)

def generate_swiss_roll(n_samples=1000):
    X, _ = make_swiss_roll(n_samples=n_samples, noise=1.5, random_state=42)
    Y, _ = make_swiss_roll(n_samples=n_samples, noise=0, random_state=42)
    return jnp.array(X), jnp.array(Y)
