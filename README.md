# Optimal Transport Estimators

This repository contains code implementations for various Optimal Transport (OT) estimators, developed as part of the Optimal Transport course at ENSAE, taught by Prof. Marco Cuturi.

We implement and compare several approaches to learn transport maps between probability distributions on synthetic datasets:

- **Neural Dual Potential (ICNN)**: Solving OT via neural convex optimization (Input Convex Neural Networks).
- **Flow Matching**: Learning the continuous velocity fields between source and target distributions.
- **Entropic OT (Sinkhorn Divergence)**: Regularized optimal transport using entropic penalization.

For each estimator, we evaluate performance on three classic synthetic datasets:

- **Gaussian Blobs**
- **Two Moons**
- **Swiss Roll**

Metrics computed include:

- **Sinkhorn Divergence** (Approximate Wasserstein distance)
- **Chamfer Distance** (Point-wise matching quality)
- **Dual Objective Value** (when available)

Visualization of the learned transport maps is provided in both 2D and 3D when appropriate.

All results are obtained with fixed random seeds for full reproducibility.

---
**Course**: Optimal Transport (ENSAE)  
**Instructor**: Marco Cuturi  
**Author**: [Your Name Here]
