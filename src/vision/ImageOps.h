#pragma once
#include "core/Types.h"
#include <array>
#include <vector>

namespace objectkey::vision
{
struct Point
{
    double x = 0, y = 0;
};
struct Rect
{
    int x = 0, y = 0, width = 0, height = 0;
    bool contains(int px, int py) const
    {
        return px >= x && py >= y && px < x + width && py < y + height;
    }
};
struct GrayImage
{
    int width = 0, height = 0;
    std::vector<float> pixels;
    GrayImage() = default;
    GrayImage(int w, int h, float value = 0);
    float &at(int x, int y)
    {
        return pixels[size_t(y) * width + x];
    }
    float at(int x, int y) const
    {
        return pixels[size_t(y) * width + x];
    }
};
struct Mask
{
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;
    Mask() = default;
    Mask(int w, int h, unsigned char value = 0);
    unsigned char &at(int x, int y)
    {
        return pixels[size_t(y) * width + x];
    }
    unsigned char at(int x, int y) const
    {
        return pixels[size_t(y) * width + x];
    }
};
struct Moments
{
    size_t area = 0;
    double cx = 0, cy = 0, xx = 0, xy = 0, yy = 0;
};
using AffineMatrix = std::array<double, 6>; // [a,b,tx,c,d,ty]
void validateImage(const Image &);
void validateRoi(const Image &, const Roi &);
Rect intersect(Rect, Rect);
Rect bounds(const Mask &);
size_t foregroundCount(const Mask &);
Moments moments(const Mask &);
Image crop(const Image &, Rect);
Mask crop(const Mask &, Rect);
Image resizeArea(const Image &, int width, int height);
Mask resizeNearest(const Mask &, int width, int height);
GrayImage grayscale(const Image &); // [0,1], BT.601 luma
GrayImage resizeBilinear(const GrayImage &, int width, int height, bool precise = false);
GrayImage gaussian(const GrayImage &, double sigma, int radius = 0);
GrayImage halfSample(const GrayImage &);
Image warpAffine(const Image &, const AffineMatrix &, int width, int height,
                 std::array<unsigned char, 3> background = {104, 116, 124});
Image rotate180(const Image &);
Mask readMask(const Image &, const Roi &);
Image maskImage(const Mask &);
std::vector<float> labChroma(const Image &);
void paintDisk(Mask &, int x, int y, int radius, unsigned char value);
void paintLine(Mask &, int x0, int y0, int x1, int y1, int radius, unsigned char value);
} // namespace objectkey::vision
