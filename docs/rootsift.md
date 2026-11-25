# RootSIFT descriptor normalization

[Back to README](../README.md#how-it-works)

RootSIFT transforms a nonnegative SIFT histogram so that descriptor comparison is less dominated by a few large bins.

## Principle

For descriptor $`d`$, first normalize its histogram mass, then take an elementwise square root:

```math
h_i=\frac{d_i}{\sum_j d_j},\qquad r_i=\sqrt{h_i}.
```

For two L1-normalized histograms, the dot product of their square-root embeddings is the Hellinger kernel:

```math
r^{T}r'=\sum_i\sqrt{h_i h'_i}.
```

This changes the geometry of histogram comparison without changing the detected keypoints.

## Implementation

ObjectKEY follows the pretrained SIFT LightGlue preprocessing by applying a small floor $`\varepsilon`$ before the square root and normalizing again:

```math
\widetilde{r}_i=\sqrt{\max(h_i,\varepsilon)},\qquad
\widehat{r}=\frac{\widetilde{r}}{\lVert\widetilde{r}\rVert_2}.
```

The function validates descriptor dimensions and values. Stored descriptors remain SIFT vectors; matcher preparation converts a copy to RootSIFT so repeated comparisons do not repeatedly transform the stored data.

**Code:** `rootSift` in [Sift.cpp](../src/vision/Sift.cpp); `LightGlue::Impl::prepare` in [LightGlue.cpp](../src/features/LightGlue.cpp).

**References:** [6](../README.md#ref-6), [2](../README.md#ref-2).
