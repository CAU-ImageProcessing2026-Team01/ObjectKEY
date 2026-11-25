# Cosine similarity and the matching decision

[Back to README](../README.md#how-it-works)

Cosine similarity compares the direction of two global [ResNet embeddings](resnet18.md). Local-feature geometry provides a separate consistency check.

## Appearance similarity

For embeddings $`a`$ and $`b`$,

```math
s(a,b)=\frac{a^{T}b}{\lVert a\rVert_2\lVert b\rVert_2}.
```

ObjectKEY L2-normalizes both vectors, so the calculation becomes

```math
s(a,b)=a^{T}b=\sum_i a_i b_i.
```

The C++ implementation validates the vectors, accumulates the dot product in double precision and clamps rounding error to the cosine range.

## Combining appearance and geometry

Let $`\mathcal{M}`$ be the LightGlue correspondence set and $`\mathcal{I}`$ its inliers under a valid [RANSAC affine model](ransac.md). Define

```math
n=\lvert\mathcal{I}\rvert,\qquad
r=\frac{\lvert\mathcal{I}\rvert}{\lvert\mathcal{M}\rvert}.
```

A comparison passes when

```math
\mathrm{Accept}=
(s\ge\tau_{\mathrm{appearance}})
\land(n\ge\tau_{\mathrm{count}})
\land(r\ge\tau_{\mathrm{ratio}}).
```

The appearance threshold requires global similarity. The count threshold requires enough consistent local evidence, while the ratio threshold requires that evidence to explain a sufficient fraction of the matches. A failed geometric fit cannot pass.

The thresholds are loaded from [matching.ini](../config/matching.ini). RANSAC's residual tolerance determines which correspondences belong to $`\mathcal{I}`$.

**Code:** `normalizeEmbedding`, `cosineSimilarity` in [Embedding.cpp](../src/features/Embedding.cpp); [Alignment.cpp](../src/features/Alignment.cpp), [MatchingConfig.cpp](../src/app/MatchingConfig.cpp).

**References:** [3](../README.md#ref-3), [10](../README.md#ref-10).
