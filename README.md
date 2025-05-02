# Optimal Transport Estimators

This repository contains code implementations for various Optimal Transport (OT) estimators, developed as part of the Optimal Transport course at ENSAE, taught by Prof. Marco Cuturi.

We implement and compare several approaches to learn transport maps between probability distributions on synthetic datasets:

- **Neural Dual Potential (ICNN)**: Solving OT via neural convex optimization with input-convex neural networks [1,2].
- **Flow Matching**: Learning continuous velocity fields to match distributions, avoiding explicit potentials [3].
- **Entropic OT (Sinkhorn Divergence)**: Regularized optimal transport using entropic penalization for efficient computation [4,5].

For each estimator, we evaluate performance on three synthetic datasets:

- **Gaussian Blobs**
- **Two Moons**
- **Swiss Roll**

Metrics computed include:

- **Sinkhorn Divergence** (Approximate Wasserstein distance)
- **Chamfer Distance** (Point-wise matching quality)
- **Dual Objective Value** (when available)

Visualization of the learned transport maps is provided in both 2D and 3D when appropriate.

All experiments are conducted with fixed random seeds for full reproducibility.


--

## References

[1] Amos, B., Xu, L., & Kolter, J. Z. (2017). *Input Convex Neural Networks*. In Proceedings of the 34th International Conference on Machine Learning (ICML).

[2] Makkuva, A., Taghvaei, A., Oh, S., & Yang, J. (2020). *Optimal Transport Mapping via Input Convex Neural Networks*. In Proceedings of the 37th International Conference on Machine Learning (ICML).

[3] Lipman, Y. (2023). *Flow Matching for Generative Modeling*. Advances in Neural Information Processing Systems (NeurIPS 2023).

[4] Sinkhorn, R. (1967). *Diagonal Equivalence to Matrices with Prescribed Row and Column Sums*. The American Mathematical Monthly.

[5] Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. In Advances in Neural Information Processing Systems (NeurIPS).

