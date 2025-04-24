import numpy as np
import ot

def mini_batch(data, batch_size):
    """
    Randomly sample a mini-batch from a dataset with uniform probability.

    This function selects `batch_size` elements from `data` without replacement
    and assigns uniform weights to the selected points to form a discrete probability distribution.

    Parameters
    ----------
    data : ndarray of shape (N, d)
        The input data, where N is the number of samples and d is the dimension.
    batch_size : int
        The number of samples to include in the mini-batch.

    Returns
    -------
    minibatch : ndarray of shape (batch_size, d)
        A mini-batch of sampled points.
    sub_weights : ndarray of shape (batch_size,)
        Uniform weights assigned to each point in the mini-batch.
    id_batch : ndarray of shape (batch_size,)
        Indices of the selected samples in the original dataset.
    """
    id_ = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = ot.unif(batch_size)
    return data[id_], sub_weights, id_

def update_plan(pi, pi_minibatch, id_a, id_b):
    """
    Update a global transport plan with the results from a mini-batch computation.

    The function inserts the mini-batch optimal transport plan into the appropriate
    positions in the global transport matrix.

    Parameters
    ----------
    pi : ndarray of shape (ns, nt)
        The global (accumulated) transport plan matrix.
    pi_minibatch : ndarray of shape (m, m)
        The optimal transport plan computed for the mini-batch.
    id_a : ndarray of shape (m,)
        Indices of the selected source points in the global source distribution.
    id_b : ndarray of shape (m,)
        Indices of the selected target points in the global target distribution.

    Returns
    -------
    pi : ndarray of shape (ns, nt)
        The updated global transport plan matrix.
    """
    for i, i2 in enumerate(id_a):
        for j, j2 in enumerate(id_b):
            pi[i2, j2] += pi_minibatch[i][j]
    return pi

def compute_incomplete_plan(xs, xt, a, b, bs, K, C, lambd=1e-1, method="exact"):
    """
    Estimate the optimal transport plan using stochastic mini-batches.

    This function computes an approximate OT coupling (transport plan) by averaging
    over several mini-batch computations. It supports both exact (EMD) and entropic
    regularized OT via Sinkhorn.

    Parameters
    ----------
    xs : ndarray of shape (ns, d)
        Source samples.
    xt : ndarray of shape (nt, d)
        Target samples.
    a : ndarray of shape (ns,)
        Probability weights for the source distribution.
    b : ndarray of shape (nt,)
        Probability weights for the target distribution.
    bs : int
        Mini-batch size.
    K : int
        Number of mini-batch iterations to perform.
    C : ndarray of shape (ns, nt)
        Precomputed cost matrix between all source and target samples.
    lambd : float, optional (default=1e-1)
        Entropic regularization strength (used if `method="entropic"`).
    method : str, optional (default="exact")
        OT solver method. Either:
        - "exact": exact OT using Earth Mover’s Distance (EMD)
        - "entropic": entropic OT using Sinkhorn algorithm

    Returns
    -------
    incomplete_pi : ndarray of shape (ns, nt)
        Estimated OT coupling (transport plan) averaged over K mini-batch computations.
    """
    incomplete_pi = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    for i in range(K):
        # Sample mini-batches from source and target
        sub_xs, sub_weights_a, id_a = mini_batch(xs, bs)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, bs)

        # Extract mini-batch cost submatrix
        mb_C = C[id_a, :][:, id_b]

        # Solve OT between mini-batches
        if method == "exact":
            G0 = ot.emd(sub_weights_a, sub_weights_b, mb_C.copy())
        elif method == "entropic":
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, mb_C, lambd)

        # Update global plan
        incomplete_pi = update_plan(incomplete_pi, G0, id_a, id_b)

    return (1 / K) * incomplete_pi
