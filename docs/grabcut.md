# GrabCut foreground segmentation

[Back to README](../README.md#how-it-works)

GrabCut separates an object from its background by alternating color-model estimation and graph-cut segmentation.

## Initialization and color models

Pixels have four possible states: definite background, definite foreground, probable background and probable foreground. Rectangle initialization marks the inside as probable foreground and the outside as definite background. Manual foreground/background strokes provide fixed labels.

Foreground and background each have a Gaussian mixture model (GMM). For color vector $z$ and class $\alpha$,

$$
p(z\mid\alpha)=\sum_{k=1}^{K}\pi_{\alpha k}
\mathcal{N}(z;\mu_{\alpha k},\Sigma_{\alpha k}),\qquad
D_p(\alpha)=-\log p(z_p\mid\alpha).
$$

The implementation initializes clusters with k-means++, assigns samples to weighted Gaussian components, and updates mixture weights, means and full covariance matrices. Covariance regularization and Cholesky factorization stabilize probability evaluation; log-sum-exp stabilizes the mixture likelihood.

## Segmentation energy

Let $\alpha_p$ denote a pixel's foreground/background label. GrabCut minimizes

$$
E(\alpha)=\sum_p D_p(\alpha_p)
+\sum_{(p,q)\in\mathcal{N}}w_{pq}\,[\alpha_p\ne\alpha_q],
$$

where the brackets equal one when neighboring labels differ, and zero otherwise. Neighbor weights are

$$
w_{pq}=\frac{\gamma}{\operatorname{dist}(p,q)}
\exp\left(-\beta\lVert z_p-z_q\rVert_2^2\right),\qquad
\beta=\frac{1}{2\,\mathbb{E}_{(p,q)\in\mathcal{N}}
[\lVert z_p-z_q\rVert_2^2]}.
$$

The data term favors the class whose color model explains the pixel. The boundary term discourages cuts between similarly colored neighbors. The parameter $\gamma$ controls their relative influence.

## Implementation

The graph uses eight-neighbor pixel connections. Source-side pixels represent foreground; source-to-pixel and pixel-to-sink capacities encode background and foreground costs respectively. A common offset is subtracted from each pair of data costs to keep capacities nonnegative. Large terminal capacities and fixed-label updates preserve manual constraints.

Each iteration updates the GMMs, rebuilds the graph, solves [maximum flow/minimum cut](max-flow.md), and updates probable labels. The final definite/probable foreground labels form the binary mask. GUI correction reruns the same process from the edited labels.

**Code:** [GrabCut.cpp](../src/vision/GrabCut.cpp), `makeGrabCutMask` in [Alignment.cpp](../src/features/Alignment.cpp), [GrabCutEditor.cpp](../src/gui/GrabCutEditor.cpp).

**Reference:** [11](../README.md#ref-11).
