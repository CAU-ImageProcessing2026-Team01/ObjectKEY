# LightGlue local-feature matching

[Back to README](../README.md#how-it-works)

LightGlue matches two sets of local features using learned attention. ObjectKEY supplies keypoint position, scale, orientation and [RootSIFT descriptors](rootsift.md).

## Feature interaction

The initialization network projects descriptors into a learned feature space and encodes their geometry. Subsequent layers combine:

- **Self-attention:** relationships among features within an image.
- **Cross-attention:** relationships between features in the two images.
- **Assignment scores:** the compatibility of candidate correspondences.

These contextual updates help distinguish similar local descriptors using surrounding features.

## Adaptive computation

Token confidence indicates whether a feature's matching state is sufficiently resolved. Matchability estimates whether it is likely to have a counterpart.

C++ runs the exported layer graphs sequentially. It stops when the confident-feature fraction exceeds a depth threshold, and prunes confidently unmatchable features between layers. Layer-dependent confidence thresholds control these decisions.

## Correspondence selection

Let $S_{ij}$ be the predicted log assignment score. A pair is retained when it is the mutual best assignment and exceeds a match-score threshold:

$$
j=\operatorname*{arg\,max}_{k}S_{ik},\qquad
i=\operatorname*{arg\,max}_{k}S_{kj},\qquad
\exp(S_{ij})>\tau_{\mathrm{match}}.
$$

The implementation maps retained indexes back to the original feature arrays. These one-to-one correspondences are then passed to [affine RANSAC](ransac.md).

## Implementation

The official SIFT checkpoint is exported into an initialization graph and separate layer graphs. ONNX Runtime executes the pretrained network; C++ handles adaptive stopping, pruning and final pair selection.

**Code:** [LightGlue.cpp](../src/features/LightGlue.cpp), [export-lightglue.py](../config/setup/export-lightglue.py).

**Reference:** [2](../README.md#ref-2).
