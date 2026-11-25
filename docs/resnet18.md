# ResNet-18 appearance features

[Back to README](../README.md#how-it-works)

ResNet-18 summarizes object appearance as a global feature vector. ObjectKEY uses Torchvision's ImageNet-pretrained backbone with the final classifier removed.

## Residual features

A residual block combines a learned transformation with a shortcut:

```math
y=\mathrm{ReLU}\bigl(F(x)+P(x)\bigr).
```

Here, $`P`$ is an identity mapping or a projection when feature dimensions change. Stacked residual blocks extract increasingly abstract visual features. Global average pooling produces the 512-dimensional vector used by ObjectKEY.

## Image normalization

[Preprocessing](preprocessing.md) provides a foreground-aligned $`224\times224`$ image with a neutral background. Pixel values are converted from BGR to RGB and scaled to the unit interval. Each channel uses the pretrained model's mean $`\mu_c`$ and standard deviation $`\sigma_c`$:

```math
X_c(x,y)=\frac{I_c(x,y)-\mu_c}{\sigma_c}.
```

The tensor is arranged in channel-first order and evaluated through ONNX Runtime.

## Pooling and L2 normalization

Define unit-length normalization as

```math
\mathrm{norm}(v)=\frac{v}{\lVert v\rVert_2}.
```

The mask's principal axis can point in either direction. ObjectKEY extracts features from the aligned image $`I`$ and its half-turn rotation $`R_\pi I`$, then pools them:

```math
e=\mathrm{norm}\left(
\mathrm{norm}(f(I))+
\mathrm{norm}(f(R_\pi I))
\right).
```

The resulting embedding is compared with the reference embedding using [cosine similarity](cosine-similarity.md).

## Implementation

The exporter replaces the fully connected classifier with an identity operation while retaining global average pooling. C++ prepares the image tensor, runs the exported ONNX graph and performs normalization/pooling with explicit vector operations.

**Code:** [ResNet18.cpp](../src/features/ResNet18.cpp), [Embedding.cpp](../src/features/Embedding.cpp), [RecommendedPipeline.cpp](../src/features/RecommendedPipeline.cpp), [export-resnet18.py](../config/setup/export-resnet18.py).

**References:** [3](../README.md#ref-3), [17](../README.md#ref-17), [19](../README.md#ref-19).
