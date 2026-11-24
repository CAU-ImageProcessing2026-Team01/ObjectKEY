# Image preprocessing and ROI handling

[Back to README](../README.md#how-it-works)

Preprocessing isolates the selected object and prepares separate inputs for global appearance and local-feature extraction.

## ROI and foreground mask

The user selects a rectangular region of interest (ROI). [GrabCut](grabcut.md) initializes its inside as probable foreground and its outside as background, then estimates a binary foreground mask. A supplied or manually corrected mask can replace this segmentation.

The implementation resizes the image for segmentation, maps the ROI into that canvas, and preserves an outer background border. It maps the resulting mask back to the original image using nearest-neighbor sampling. The foreground bounding box, with padding clipped to the ROI, defines a shared object-centered crop.

## Appearance input

Foreground pixels determine the centroid and central moments:

$$
\bar{x}=\frac{\sum_{x,y}xM(x,y)}{\sum_{x,y}M(x,y)},\qquad
\bar{y}=\frac{\sum_{x,y}yM(x,y)}{\sum_{x,y}M(x,y)}
$$

$$
\mu_{pq}=\sum_{x,y}M(x,y)(x-\bar{x})^p(y-\bar{y})^q,\qquad
\theta=\frac{1}{2}\operatorname{atan2}(2\mu_{11},\mu_{20}-\mu_{02})
$$

Here, $M$ is the binary mask. The principal-axis angle $\theta$ provides a common orientation for the object. The implementation aligns that axis vertically, fits the foreground into the ResNet input canvas, and replaces the background with a neutral color. Inverse affine mapping and bilinear sampling construct the aligned image.

A principal axis has two possible directions. [ResNet feature pooling](resnet18.md) combines the aligned image and its half-turn rotation to handle this ambiguity.

## Local-feature input

The local branch resizes the original crop while preserving its aspect ratio, converts it to grayscale, and enlarges it before [DoG/SIFT extraction](dog-keypoints.md). It maps detected coordinates back to the local canvas, removes repeated position/scale entries, and keeps keypoints whose centers lie inside the foreground mask.

This branch retains the original crop pixels. The mask selects keypoint centers; descriptor neighborhoods near the object boundary can still contain background.

## Supporting implementation

| Operation | Implementation |
| --- | --- |
| Image decoding | Windows WIC into owned BGR pixel arrays |
| Color-image resizing | Area integration over source pixel footprints |
| Gray-image interpolation and warping | Bilinear sampling |
| Binary-mask resizing | Nearest-neighbor sampling |
| Orientation alignment | Foreground moments and a planar rotation |
| Scale-space smoothing | [Separable Gaussian convolution](gaussian-filtering.md) |

**Code:** [RecommendedPipeline.cpp](../src/features/RecommendedPipeline.cpp), [Alignment.cpp](../src/features/Alignment.cpp), [ImageOps.cpp](../src/vision/ImageOps.cpp), [ImageLoader.cpp](../src/io/ImageLoader.cpp).

**Reference:** [8](../README.md#ref-8).
