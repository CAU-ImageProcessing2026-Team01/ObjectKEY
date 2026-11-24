#pragma once
#include "vision/ImageOps.h"
namespace objectkey::vision
{
struct SiftOptions
{
    unsigned maximumFeatures = 500;
    int layers = 3;
    double contrast = .01, edge = 10, sigma = 1.6;
};
// Input is luminance in [0,1]. Output coordinates/diameters use input-image pixels.
FeatureSet sift(const GrayImage &, const SiftOptions &options = {});
// LightGlue-compatible RootSIFT, applied independently to every 128-vector.
void rootSift(std::vector<float> &descriptors);
} // namespace objectkey::vision
