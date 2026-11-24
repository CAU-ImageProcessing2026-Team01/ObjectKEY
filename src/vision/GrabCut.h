#pragma once
#include "vision/ImageOps.h"
namespace objectkey::vision
{
enum GrabLabel : unsigned char
{
    Background = 0,
    Foreground = 1,
    ProbableBackground = 2,
    ProbableForeground = 3
};
Mask rectangleLabels(int width, int height, Rect rectangle);
Mask foregroundMask(const Mask &labels);
// Five-component full-covariance GMMs and iterative 8-neighbor graph cuts.
// Hard labels (0/1) remain fixed; only probable labels (2/3) may change.
Mask grabCut(const Image &, Mask labels, unsigned iterations = 5);
} // namespace objectkey::vision
