# SIFT orientation and descriptors

[Back to README](../README.md#how-it-works)

SIFT describes the gradient pattern around a localized [DoG keypoint](dog-keypoints.md). It assigns an orientation and builds a 128-dimensional descriptor.

## Orientation assignment

For Gaussian-smoothed image $`L`$, central differences give the gradient:

```math
g_x=L(x+1,y)-L(x-1,y),\qquad
g_y=L(x,y+1)-L(x,y-1),
```

```math
m=\sqrt{g_x^2+g_y^2},\qquad
\phi=\mathrm{atan2}(g_y,g_x).
```

Gradient magnitudes accumulate into an orientation histogram with Gaussian spatial weighting. The implementation smooths the circular histogram and interpolates its peaks. Strong secondary peaks above a relative peak threshold create additional orientations.

## Descriptor construction

Sampling coordinates are rotated into the keypoint's orientation and scaled by its detected scale. The neighborhood is divided into spatial cells, each containing a gradient-direction histogram:

```math
4\times4\times8=128.
```

A Gaussian window emphasizes nearby samples. Trilinear interpolation distributes each contribution across neighboring spatial cells and orientation bins.

For raw histogram $`h`$, normalization and clipping are

```math
d_1=\frac{h}{\lVert h\rVert_2},\qquad
\widetilde{d}_i=\min(d_{1,i},\tau_{\mathrm{clip}}),\qquad
d_2=\frac{\widetilde{d}}{\lVert\widetilde{d}\rVert_2}.
```

Clipping reduces the influence of unusually large gradients.

## Implementation

The descriptor uses the image-coordinate orientation convention expected by SIFT LightGlue. It applies conventional SIFT quantization and restores unit length before storing each feature's position, scale, orientation and descriptor.

Keypoint detection and descriptor construction are separate steps orchestrated by the `sift` function. [RootSIFT](rootsift.md) transforms the stored descriptors when preparing them for [LightGlue](lightglue.md).

**Code:** `orientations`, `descriptor`, `sift` in [Sift.cpp](../src/vision/Sift.cpp); `extractLocal` in [Alignment.cpp](../src/features/Alignment.cpp).

**Reference:** [5](../README.md#ref-5).
