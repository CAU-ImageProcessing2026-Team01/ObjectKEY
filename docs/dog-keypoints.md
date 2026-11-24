# DoG keypoint detection

[Back to README](../README.md#how-it-works)

Difference of Gaussians (DoG) identifies local image structures that remain distinctive across scale. The detector returns localized positions and scales for [SIFT descriptors](sift.md).

## Scale space

For grayscale image $I$, the Gaussian scale space and its differences are

$$
L(x,y,\sigma)=G_\sigma*I,\qquad
D(x,y,\sigma)=L(x,y,k\sigma)-L(x,y,\sigma).
$$

With $s$ intervals per octave, $k=2^{1/s}$. Each octave contains progressively blurred images; the next octave downsamples a level whose scale has doubled. [Separable Gaussian filtering](gaussian-filtering.md) constructs each level using the required additional blur.

## Detection and localization

A candidate must be a strict maximum or minimum among its 26 neighbors across position and adjacent scale levels.

Around a candidate, approximate the DoG response by

$$
D(\delta)\approx D+\mathbf{g}^{T}\delta+
\frac{1}{2}\delta^{T}H\delta,\qquad
H\delta=-\mathbf{g}.
$$

The gradient $\mathbf{g}$ and Hessian $H$ use central differences over image position and scale. Partial-pivot Gaussian elimination solves for the offset $\delta$. If necessary, localization moves to the nearest sample and repeats.

The interpolated response

$$
\widehat{D}=D+\frac{1}{2}\mathbf{g}^{T}\delta
$$

must exceed a contrast threshold in magnitude after scale-interval normalization. Edge-like responses are rejected using the spatial Hessian $H_{xy}$:

$$
\det(H_{xy})>0,\qquad
\frac{\operatorname{tr}(H_{xy})^2}{\det(H_{xy})}
<\frac{(r+1)^2}{r},
$$

where $r$ controls the allowed principal-curvature ratio.

## Implementation

The detector begins with an enlarged image and compensates for its assumed initial blur. It searches interior pyramid levels, refines extrema, assigns orientations and ranks candidates by response. A feature budget limits the retained set.

The application maps coordinates back to its local canvas, removes repeated position/scale entries and keeps foreground-centered keypoints. Orientation assignment can produce several candidates at one detected extremum.

**Code:** `sift`, `localize` in [Sift.cpp](../src/vision/Sift.cpp); `solve3` in [Affine.cpp](../src/vision/Affine.cpp); coordinate mapping in [Alignment.cpp](../src/features/Alignment.cpp).

**Reference:** [5](../README.md#ref-5).
