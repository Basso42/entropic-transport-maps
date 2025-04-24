from jax import jit
import jax
from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud
import jax.numpy as jnp

@jax.jit
def sinkhorn_loss(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    epsilon: float = 0.1
) -> float:
    """
    Computes the Sinkhorn divergence between two empirical distributions.

    This function measures the distance between two batches of samples `x` and `y`
    using the entropically regularized optimal transport (Sinkhorn) divergence.

    Parameters
    ----------
    x : jnp.ndarray of shape (n, d)
        First batch of samples, representing the source distribution.
    y : jnp.ndarray of shape (m, d)
        Second batch of samples, representing the target distribution.
    epsilon : float, optional (default=0.1)
        Entropic regularization strength. Lower values approximate the true
        optimal transport cost more closely but can be less stable.

    Returns
    -------
    divergence : float
        The Sinkhorn divergence between the empirical distributions defined by `x` and `y`.
    """
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, epsilon=epsilon, a=a, b=b
    )
    return sdiv.divergence
