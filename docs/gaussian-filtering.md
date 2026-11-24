# Gaussian filtering

[Back to README](../README.md#how-it-works)

Gaussian filtering suppresses fine-scale intensity changes. In ObjectKEY, it builds the smoothed images used by the [DoG detector](dog-keypoints.md).

## Principle

For standard deviation $\sigma$ and kernel radius $r$, the normalized one-dimensional kernel is

$$
g_\sigma(i)=
\frac{\exp\left(-i^2/(2\sigma^2)\right)}
{\sum_{j=-r}^{r}\exp\left(-j^2/(2\sigma^2)\right)}.
$$

The two-dimensional Gaussian is separable:

$$
G_\sigma(x,y)=g_\sigma(x)g_\sigma(y),\qquad
L=G_\sigma*I.
$$

Applying horizontal and vertical convolution therefore produces the same result as the corresponding two-dimensional kernel. For kernel width $k$, separability reduces work per pixel from $O(k^2)$ to $O(k)$.

## Implementation

The C++ implementation constructs and normalizes the kernel, convolves each row into a temporary image, then convolves each column. Out-of-bounds samples reflect across the boundary without repeating the edge pixel. Intermediate values use floating-point storage.

Gaussian variances add under convolution. To move from an existing blur $\sigma_1$ to a larger blur $\sigma_2$, the scale-space builder applies

$$
\sigma_{\mathrm{additional}}=\sqrt{\sigma_2^2-\sigma_1^2}.
$$

Subtracting neighboring Gaussian levels then forms the DoG pyramid.

**Code:** `gaussian` in [ImageOps.cpp](../src/vision/ImageOps.cpp); pyramid construction in [Sift.cpp](../src/vision/Sift.cpp).

**References:** [5](../README.md#ref-5), [8](../README.md#ref-8).
