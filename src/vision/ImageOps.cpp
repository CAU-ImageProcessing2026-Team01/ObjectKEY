#include "vision/ImageOps.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace objectkey::vision
{
namespace
{
size_t area(int w, int h)
{
    if (w <= 0 || h <= 0 || uint64_t(w) * h > 160000000)
        throw std::runtime_error("Invalid image dimensions.");
    return size_t(w) * h;
}
int reflect(int x, int n)
{
    if (n == 1)
        return 0;
    while (x < 0 || x >= n)
        x = x < 0 ? -x : 2 * n - 2 - x;
    return x;
}
unsigned char byte(double x)
{
    return static_cast<unsigned char>(std::clamp(std::lround(x), 0L, 255L));
}
} // namespace
GrayImage::GrayImage(int w, int h, float value) : width(w), height(h), pixels(area(w, h), value) {}
Mask::Mask(int w, int h, unsigned char value) : width(w), height(h), pixels(area(w, h), value) {}
void validateImage(const Image &i)
{
    if (!i.width || !i.height || uint64_t(i.width) * i.height > 40000000 ||
        i.pixels.size() != size_t(i.width) * i.height * 3)
        throw std::runtime_error("Invalid BGR image buffer.");
}
void validateRoi(const Image &i, const Roi &r)
{
    validateImage(i);
    if (!r.width || !r.height || r.x >= i.width || r.y >= i.height || r.width > i.width - r.x ||
        r.height > i.height - r.y)
        throw std::runtime_error("Invalid ROI.");
}
Rect intersect(Rect a, Rect b)
{
    int x = std::max(a.x, b.x), y = std::max(a.y, b.y);
    return {x, y, std::max(0, std::min(a.x + a.width, b.x + b.width) - x),
            std::max(0, std::min(a.y + a.height, b.y + b.height) - y)};
}
Rect bounds(const Mask &m)
{
    int x0 = m.width, y0 = m.height, x1 = -1, y1 = -1;
    for (int y = 0; y < m.height; y++)
        for (int x = 0; x < m.width; x++)
            if (m.at(x, y))
            {
                x0 = std::min(x0, x);
                y0 = std::min(y0, y);
                x1 = std::max(x1, x);
                y1 = std::max(y1, y);
            }
    return x1 < 0 ? Rect{} : Rect{x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}
size_t foregroundCount(const Mask &m)
{
    return size_t(std::count_if(m.pixels.begin(), m.pixels.end(), [](auto v) { return v != 0; }));
}
Moments moments(const Mask &m)
{
    Moments r;
    for (int y = 0; y < m.height; y++)
        for (int x = 0; x < m.width; x++)
            if (m.at(x, y))
            {
                r.area++;
                r.cx += x;
                r.cy += y;
            }
    if (!r.area)
        return r;
    r.cx /= r.area;
    r.cy /= r.area;
    for (int y = 0; y < m.height; y++)
        for (int x = 0; x < m.width; x++)
            if (m.at(x, y))
            {
                double dx = x - r.cx, dy = y - r.cy;
                r.xx += dx * dx;
                r.xy += dx * dy;
                r.yy += dy * dy;
            }
    return r;
}
Image crop(const Image &image, Rect r)
{
    if (r.x < 0 || r.y < 0 || r.width <= 0 || r.height <= 0)
        throw std::runtime_error("Invalid crop.");
    validateRoi(image, {unsigned(r.x), unsigned(r.y), unsigned(r.width), unsigned(r.height)});
    Image out{unsigned(r.width), unsigned(r.height), std::vector<unsigned char>(area(r.width, r.height) * 3)};
    for (int y = 0; y < r.height; y++)
        std::copy_n(image.pixels.begin() + (size_t(y + r.y) * image.width + r.x) * 3, size_t(r.width) * 3,
                    out.pixels.begin() + size_t(y) * r.width * 3);
    return out;
}
Mask crop(const Mask &mask, Rect r)
{
    if (r.x < 0 || r.y < 0 || r.width <= 0 || r.height <= 0 || r.x + r.width > mask.width ||
        r.y + r.height > mask.height)
        throw std::runtime_error("Invalid mask crop.");
    Mask out(r.width, r.height);
    for (int y = 0; y < r.height; y++)
        std::copy_n(mask.pixels.begin() + size_t(y + r.y) * mask.width + r.x, r.width,
                    out.pixels.begin() + size_t(y) * r.width);
    return out;
}
Image resizeArea(const Image &input, int w, int h)
{
    validateImage(input);
    Image out{unsigned(w), unsigned(h), std::vector<unsigned char>(area(w, h) * 3)};
    if (unsigned(w) == input.width && unsigned(h) == input.height)
        return input;
    const double sx = double(input.width) / w, sy = double(input.height) / h;
    // Integrate the piecewise-constant source over each destination footprint.
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double left = x * sx, right = (x + 1) * sx, top = y * sy, bottom = (y + 1) * sy, sum[3]{};
            for (int iy = int(top); iy < std::min(int(input.height), int(std::ceil(bottom))); iy++)
            {
                double wy = std::min(bottom, double(iy + 1)) - std::max(top, double(iy));
                for (int ix = int(left); ix < std::min(int(input.width), int(std::ceil(right))); ix++)
                {
                    double weight = wy * (std::min(right, double(ix + 1)) - std::max(left, double(ix)));
                    size_t i = (size_t(iy) * input.width + ix) * 3;
                    for (int c = 0; c < 3; c++)
                        sum[c] += weight * input.pixels[i + c];
                }
            }
            for (int c = 0; c < 3; c++)
                out.pixels[(size_t(y) * w + x) * 3 + c] = byte(sum[c] / (sx * sy));
        }
    return out;
}
Mask resizeNearest(const Mask &input, int w, int h)
{
    Mask out(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out.at(x, y) = input.at(int(int64_t(x) * input.width / w), int(int64_t(y) * input.height / h));
    return out;
}
GrayImage grayscale(const Image &input)
{
    validateImage(input);
    GrayImage out(int(input.width), int(input.height));
    for (size_t i = 0; i < out.pixels.size(); i++)
        out.pixels[i] = float(
            (.114 * input.pixels[3 * i] + .587 * input.pixels[3 * i + 1] + .299 * input.pixels[3 * i + 2]) /
            255.);
    return out;
}
GrayImage resizeBilinear(const GrayImage &in, int w, int h, bool precise)
{
    GrayImage out(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double sx = std::clamp(precise ? double(x) * in.width / w : (x + .5) * in.width / w - .5, 0.,
                                   double(in.width - 1));
            double sy = std::clamp(precise ? double(y) * in.height / h : (y + .5) * in.height / h - .5, 0.,
                                   double(in.height - 1));
            int x0 = int(sx), y0 = int(sy), x1 = std::min(x0 + 1, in.width - 1),
                y1 = std::min(y0 + 1, in.height - 1);
            double fx = sx - x0, fy = sy - y0;
            out.at(x, y) = float((1 - fy) * ((1 - fx) * in.at(x0, y0) + fx * in.at(x1, y0)) +
                                 fy * ((1 - fx) * in.at(x0, y1) + fx * in.at(x1, y1)));
        }
    return out;
}
GrayImage gaussian(const GrayImage &in, double sigma, int radius)
{
    if (!std::isfinite(sigma) || sigma <= 0)
        throw std::runtime_error("Invalid Gaussian sigma.");
    radius = radius > 0 ? radius : std::max(1, int(std::ceil(4 * sigma)));
    std::vector<double> kernel(size_t(radius) * 2 + 1);
    double sum = 0;
    for (int i = -radius; i <= radius; i++)
        sum += (kernel[i + radius] = std::exp(-i * i / (2 * sigma * sigma)));
    for (auto &v : kernel)
        v /= sum;
    GrayImage temp(in.width, in.height), out(in.width, in.height);
    for (int y = 0; y < in.height; y++)
        for (int x = 0; x < in.width; x++)
        {
            double v = 0;
            for (int k = -radius; k <= radius; k++)
                v += kernel[k + radius] * in.at(reflect(x + k, in.width), y);
            temp.at(x, y) = float(v);
        }
    for (int y = 0; y < in.height; y++)
        for (int x = 0; x < in.width; x++)
        {
            double v = 0;
            for (int k = -radius; k <= radius; k++)
                v += kernel[k + radius] * temp.at(x, reflect(y + k, in.height));
            out.at(x, y) = float(v);
        }
    return out;
}
GrayImage halfSample(const GrayImage &in)
{
    GrayImage out(in.width / 2, in.height / 2);
    for (int y = 0; y < out.height; y++)
        for (int x = 0; x < out.width; x++)
            out.at(x, y) = in.at(2 * x, 2 * y);
    return out;
}
Image warpAffine(const Image &in, const AffineMatrix &a, int w, int h, std::array<unsigned char, 3> bg)
{
    validateImage(in);
    double det = a[0] * a[4] - a[1] * a[3];
    if (!std::isfinite(det) || std::abs(det) < 1e-12)
        throw std::runtime_error("Singular affine warp.");
    Image out{unsigned(w), unsigned(h), std::vector<unsigned char>(area(w, h) * 3)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double dx = x - a[2], dy = y - a[5], sx = (a[4] * dx - a[1] * dy) / det,
                   sy = (-a[3] * dx + a[0] * dy) / det;
            int ix = int(std::floor(sx)), iy = int(std::floor(sy));
            double fx = sx - ix, fy = sy - iy;
            for (int c = 0; c < 3; c++)
            {
                auto pixel = [&](int xx, int yy) {
                    return xx < 0 || yy < 0 || xx >= int(in.width) || yy >= int(in.height)
                               ? double(bg[c])
                               : double(in.pixels[(size_t(yy) * in.width + xx) * 3 + c]);
                };
                out.pixels[(size_t(y) * w + x) * 3 + c] =
                    byte((1 - fy) * ((1 - fx) * pixel(ix, iy) + fx * pixel(ix + 1, iy)) +
                         fy * ((1 - fx) * pixel(ix, iy + 1) + fx * pixel(ix + 1, iy + 1)));
            }
        }
    return out;
}
Image rotate180(const Image &input)
{
    validateImage(input);
    Image out = input;
    size_t n = size_t(input.width) * input.height;
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++)
            out.pixels[3 * i + c] = input.pixels[3 * (n - 1 - i) + c];
    return out;
}
Mask readMask(const Image &input, const Roi &roi)
{
    validateRoi(input, roi);
    Mask out(int(roi.width), int(roi.height));
    for (unsigned y = 0; y < roi.height; y++)
        for (unsigned x = 0; x < roi.width; x++)
        {
            size_t i = (size_t(y + roi.y) * input.width + x + roi.x) * 3;
            if (input.pixels[i] != input.pixels[i + 1] || input.pixels[i + 1] != input.pixels[i + 2])
                throw std::runtime_error("Object mask must be grayscale.");
            out.at(int(x), int(y)) = input.pixels[i] > 127 ? 1 : 0;
        }
    return out;
}
Image maskImage(const Mask &mask)
{
    Image out{unsigned(mask.width), unsigned(mask.height),
              std::vector<unsigned char>(mask.pixels.size() * 3)};
    for (size_t i = 0; i < mask.pixels.size(); i++)
        for (int c = 0; c < 3; c++)
            out.pixels[3 * i + c] = mask.pixels[i] ? 255 : 0;
    return out;
}
std::vector<float> labChroma(const Image &in)
{
    validateImage(in);
    std::vector<float> out(size_t(in.width) * in.height * 2);
    auto linear = [](double s) {
        s /= 255.;
        return s <= .04045 ? s / 12.92 : std::pow((s + .055) / 1.055, 2.4);
    };
    auto f = [](double v) { return v > 216. / 24389 ? std::cbrt(v) : (24389. / 27 * v + 16) / 116; };
    for (size_t i = 0; i < out.size() / 2; i++)
    {
        double b = linear(in.pixels[3 * i]), g = linear(in.pixels[3 * i + 1]),
               r = linear(in.pixels[3 * i + 2]);
        double x = f((.4124564 * r + .3575761 * g + .1804375 * b) / .95047),
               y = f(.2126729 * r + .7151522 * g + .0721750 * b),
               z = f((.0193339 * r + .1191920 * g + .9503041 * b) / 1.08883);
        out[2 * i] = float(std::clamp(500 * (x - y), -128., 127.));
        out[2 * i + 1] = float(std::clamp(200 * (y - z), -128., 127.));
    }
    return out;
}
void paintDisk(Mask &m, int x, int y, int r, unsigned char value)
{
    for (int yy = std::max(0, y - r); yy < std::min(m.height, y + r + 1); yy++)
        for (int xx = std::max(0, x - r); xx < std::min(m.width, x + r + 1); xx++)
            if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r)
                m.at(xx, yy) = value;
}
void paintLine(Mask &m, int x0, int y0, int x1, int y1, int r, unsigned char v)
{
    int n = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int i = 0; i <= n; i++)
    {
        double t = n ? double(i) / n : 0;
        paintDisk(m, int(std::lround(x0 + t * (x1 - x0))), int(std::lround(y0 + t * (y1 - y0))), r, v);
    }
}
} // namespace objectkey::vision
