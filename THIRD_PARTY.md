# Third-party components

ObjectKEY uses these upstream components. Setup downloads them locally; third-party SDKs, source checkouts and weights are not vendored in this repository.

| Component | Source | Usage |
| --- | --- | --- |
| ONNX Runtime 1.20.1 | https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1 | CPU inference |
| ResNet-18 / Torchvision | https://github.com/pytorch/vision | Official ImageNet-1K V1 weights; final classifier removed |
| SIFT LightGlue | https://github.com/cvg/LightGlue/tree/5a650c444718caffc2669dc0094cde908f42dd73 | Pinned model definition and SIFT checkpoint exported to ONNX |

Preserve applicable upstream licenses and third-party notices when distributing binaries or models. Setup retains ONNX Runtime's `LICENSE` (MIT) and `ThirdPartyNotices.txt`, and LightGlue's `LICENSE-LightGlue` (Apache-2.0) with the exported matcher. Torchvision software is BSD-3-Clause. Software licenses do not automatically establish the license of training datasets or every pretrained weight; refer to upstream model terms and attribution.

The C++ application no longer uses OpenCV. DoG/SIFT, RootSIFT, affine fitting/RANSAC, GrabCut, maximum flow, and pixel operations are project-owned implementations based on the algorithm references in [README.md](README.md#references). ONNX Runtime still executes the pretrained ResNet and LightGlue networks.

The LightGlue exporter verifies model source SHA-256 `dcb75b9cad1985c5e7a537c512d4318d47fbcc99df75b39a513de60e027c039f` and checkpoint SHA-256 `5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`. The resulting `models/lightglue_sift/provenance.json` records every graph's fingerprint. The native loader verifies these graphs before inference.
