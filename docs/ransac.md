# Affine RANSAC and inlier verification

[Back to README](../README.md#how-it-works)

RANSAC estimates a geometric model even when some feature correspondences are incorrect. ObjectKEY fits an [affine transformation](affine-transform.md) to [LightGlue matches](lightglue.md).

## Hypotheses and inliers

Each hypothesis samples three distinct, non-collinear correspondences and fits an affine model. Reflections and near-singular transforms are rejected.

For reference point $p_i$ and query point $q_i$, the forward residual is

$$
e_i=\lVert Ap_i+t-q_i\rVert_2.
$$

A correspondence is an inlier when $e_i\le\tau_{\mathrm{residual}}$. ObjectKEY scales this tolerance by the query foreground bounding-box diagonal:

$$
\tau_{\mathrm{residual}}=\eta\sqrt{w_q^2+h_q^2}.
$$

Here, $\eta$ is the configured residual ratio, and $w_q,h_q$ are measured in the query's local-feature canvas.

The winning hypothesis has the most inliers. Ties are resolved by the truncated squared-error loss

$$
L=\sum_i\min(e_i^2,\tau_{\mathrm{residual}}^2).
$$

## Adaptive iteration count

For estimated inlier fraction $w$, target sampling confidence $p$ and sample size $m$, the required number of independent hypotheses is

$$
N=\left\lceil\frac{\log(1-p)}{\log(1-w^m)}\right\rceil.
$$

The affine solver uses $m=3$. The implementation caps the iteration budget and updates it when a better model is found. A fixed random seed makes repeated evaluations reproducible.

## Refinement and verification

The winning model is refitted to all its inliers with Householder QR. Inliers are reclassified, and refinement repeats until the set stabilizes or the refinement limit is reached.

Geometry passes when the inlier count and ratio exceed their thresholds:

$$
\lvert\mathcal{I}\rvert\ge\tau_{\mathrm{count}},\qquad
\frac{\lvert\mathcal{I}\rvert}{\lvert\mathcal{M}\rvert}
\ge\tau_{\mathrm{ratio}}.
$$

Final acceptance also requires [ResNet cosine similarity](cosine-similarity.md). Because residuals are measured from reference to query and use the query's scale, reversing the pair can produce different results.

**Code:** `ransacAffine` in [Affine.cpp](../src/vision/Affine.cpp); `checkObjectMatches` in [Alignment.cpp](../src/features/Alignment.cpp).

**Reference:** [7](../README.md#ref-7).
