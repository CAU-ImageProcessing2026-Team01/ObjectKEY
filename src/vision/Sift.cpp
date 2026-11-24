#include "vision/Sift.h"
#include "vision/Affine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace objectkey::vision
{
namespace
{
constexpr double pi = std::numbers::pi;
struct Candidate
{
    int octave = 0, layer = 0;
    double x = 0, y = 0, sigma = 0, response = 0, angle = 0;
};
using Pyramid = std::vector<std::vector<GrayImage>>;
double wrap(double a)
{
    a = std::fmod(a, 2 * pi);
    return a < 0 ? a + 2 * pi : a;
}
bool localize(const std::vector<GrayImage> &d, int x, int y, int layer, const SiftOptions &cfg,
              Candidate &result)
{
    std::array<double, 3> delta{}, gradient{};
    bool converged = false;
    double dxx = 0, dyy = 0, dxy = 0;
    for (int step = 0; step < 5; step++)
    {
        if (layer < 1 || layer > cfg.layers || x < 5 || y < 5 || x >= d[0].width - 5 || y >= d[0].height - 5)
            return false;
        const auto &a = d[layer];
        double center = a.at(x, y);
        gradient = {.5 * (a.at(x + 1, y) - a.at(x - 1, y)), .5 * (a.at(x, y + 1) - a.at(x, y - 1)),
                    .5 * (d[layer + 1].at(x, y) - d[layer - 1].at(x, y))};
        dxx = a.at(x + 1, y) + a.at(x - 1, y) - 2 * center;
        dyy = a.at(x, y + 1) + a.at(x, y - 1) - 2 * center;
        double dss = d[layer + 1].at(x, y) + d[layer - 1].at(x, y) - 2 * center;
        dxy = .25 * (a.at(x + 1, y + 1) - a.at(x - 1, y + 1) - a.at(x + 1, y - 1) + a.at(x - 1, y - 1));
        double dxs = .25 * (d[layer + 1].at(x + 1, y) - d[layer + 1].at(x - 1, y) -
                            d[layer - 1].at(x + 1, y) + d[layer - 1].at(x - 1, y));
        double dys = .25 * (d[layer + 1].at(x, y + 1) - d[layer + 1].at(x, y - 1) -
                            d[layer - 1].at(x, y + 1) + d[layer - 1].at(x, y - 1));
        if (!solve3({dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss}, {-gradient[0], -gradient[1], -gradient[2]},
                    delta))
            return false;
        if (std::abs(delta[0]) < .5 && std::abs(delta[1]) < .5 && std::abs(delta[2]) < .5)
        {
            converged = true;
            break;
        }
        if (std::abs(delta[0]) > 1000 || std::abs(delta[1]) > 1000 || std::abs(delta[2]) > 1000)
            return false;
        x += int(std::round(delta[0]));
        y += int(std::round(delta[1]));
        layer += int(std::round(delta[2]));
    }
    if (!converged)
        return false;
    double value =
        d[layer].at(x, y) + .5 * (gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2]);
    if (std::abs(value) * cfg.layers < cfg.contrast)
        return false;
    double det = dxx * dyy - dxy * dxy, trace = dxx + dyy;
    if (det <= 0 || trace * trace * cfg.edge >= det * (cfg.edge + 1) * (cfg.edge + 1))
        return false;
    result.layer = layer;
    result.x = x + delta[0];
    result.y = y + delta[1];
    result.sigma = cfg.sigma * std::pow(2., (layer + delta[2]) / cfg.layers);
    result.response = std::abs(value);
    return true;
}
std::vector<double> orientations(const GrayImage &image, const Candidate &p)
{
    constexpr int bins = 36;
    std::array<double, bins> hist{}, smooth{};
    const int cx = int(std::round(p.x)), cy = int(std::round(p.y)), radius = int(std::round(4.5 * p.sigma));
    double variance = 2 * std::pow(1.5 * p.sigma, 2);
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int x = cx + dx, y = cy + dy;
            if (x <= 0 || y <= 0 || x >= image.width - 1 || y >= image.height - 1)
                continue;
            double gx = image.at(x + 1, y) - image.at(x - 1, y), gy = image.at(x, y + 1) - image.at(x, y - 1);
            int bin = int(std::lround(wrap(std::atan2(gy, gx)) * bins / (2 * pi))) % bins;
            hist[bin] += std::hypot(gx, gy) * std::exp(-(dx * dx + dy * dy) / variance);
        }
    // Circular five-tap smoothing followed by parabolic peak interpolation.
    for (int i = 0; i < bins; i++)
        smooth[i] = (hist[(i + bins - 2) % bins] + 4 * hist[(i + bins - 1) % bins] + 6 * hist[i] +
                     4 * hist[(i + 1) % bins] + hist[(i + 2) % bins]) /
                    16;
    double peak = *std::max_element(smooth.begin(), smooth.end());
    std::vector<double> result;
    if (peak <= 0)
        return result;
    for (int i = 0; i < bins; i++)
    {
        double left = smooth[(i + bins - 1) % bins], center = smooth[i], right = smooth[(i + 1) % bins];
        if (center >= .8 * peak && center > left && center > right)
        {
            double offset = .5 * (left - right) / (left - 2 * center + right);
            result.push_back(wrap((i + offset) * 2 * pi / bins));
        }
    }
    return result;
}
std::array<float, 128> descriptor(const GrayImage &image, const Candidate &p)
{
    constexpr int cells = 4, bins = 8;
    std::array<double, 128> hist{};
    double cellWidth = 3 * p.sigma, co = std::cos(p.angle), si = std::sin(p.angle);
    int radius = std::min(int(std::round(cellWidth * std::sqrt(2.) * (cells + 1) * .5)),
                          int(std::hypot(image.width, image.height)));
    int cx = int(std::round(p.x)), cy = int(std::round(p.y));
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            double col = (co * dx + si * dy) / cellWidth, row = (-si * dx + co * dy) / cellWidth;
            double rb = row + 1.5, cb = col + 1.5;
            if (rb <= -1 || rb >= cells || cb <= -1 || cb >= cells)
                continue;
            int x = cx + dx, y = cy + dy;
            if (x <= 0 || y <= 0 || x >= image.width - 1 || y >= image.height - 1)
                continue;
            double gx = image.at(x + 1, y) - image.at(x - 1, y), gy = image.at(x, y + 1) - image.at(x, y - 1);
            // Keypoint angles are clockwise in image coordinates. Descriptor bins
            // use the conventional upward gradient axis expected by SIFT LightGlue.
            double ob = wrap(p.angle - std::atan2(gy, gx)) * bins / (2 * pi),
                   magnitude = std::hypot(gx, gy) * std::exp(-(row * row + col * col) / 8);
            int r0 = int(std::floor(rb)), c0 = int(std::floor(cb)), o0 = int(std::floor(ob));
            double fr = rb - r0, fc = cb - c0, fo = ob - o0;
            for (int dr = 0; dr < 2; dr++)
                for (int dc = 0; dc < 2; dc++)
                    for (int db = 0; db < 2; db++)
                    {
                        int rr = r0 + dr, cc = c0 + dc, oo = (o0 + db) % bins;
                        if (rr < 0 || rr >= cells || cc < 0 || cc >= cells)
                            continue;
                        hist[(rr * cells + cc) * bins + oo] +=
                            magnitude * (dr ? fr : 1 - fr) * (dc ? fc : 1 - fc) * (db ? fo : 1 - fo);
                    }
        }
    double norm = 0;
    for (double v : hist)
        norm += v * v;
    double clip = .2 * std::sqrt(norm);
    norm = 0;
    for (auto &v : hist)
    {
        v = std::min(v, clip);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    std::array<float, 128> result{};
    if (norm < 1e-12)
        return result;
    // Match the conventional 512-scaled SIFT quantization used by the checkpoint.
    double qnorm = 0;
    for (size_t i = 0; i < 128; i++)
    {
        result[i] = float(std::clamp(std::lround(512 * hist[i] / norm), 0L, 255L));
        qnorm += result[i] * result[i];
    }
    qnorm = std::sqrt(qnorm);
    if (qnorm > 0)
        for (auto &v : result)
            v = float(v / qnorm);
    return result;
}
} // namespace
FeatureSet sift(const GrayImage &input, const SiftOptions &cfg)
{
    if (input.width < 8 || input.height < 8 || cfg.layers < 1 || cfg.layers > 8 || cfg.maximumFeatures == 0 ||
        cfg.sigma <= 1 || cfg.contrast <= 0 || cfg.edge <= 0)
        throw std::runtime_error("Invalid SIFT input or settings.");
    for (float v : input.pixels)
        if (!std::isfinite(v) || v < 0 || v > 1.001f)
            throw std::runtime_error("SIFT luminance must be in [0,1].");
    auto base = gaussian(resizeBilinear(input, input.width * 2, input.height * 2, true),
                         std::sqrt(cfg.sigma * cfg.sigma - 1.));
    Pyramid gaussians;
    std::vector<Candidate> candidates;
    const double k = std::pow(2., 1. / cfg.layers);
    for (int octave = -1; std::min(base.width, base.height) >= 16; octave++)
    {
        std::vector<GrayImage> g;
        g.push_back(std::move(base));
        for (int s = 1; s < cfg.layers + 3; s++)
        {
            double previous = cfg.sigma * std::pow(k, s - 1), next = previous * k;
            g.push_back(gaussian(g.back(), std::sqrt(next * next - previous * previous)));
        }
        std::vector<GrayImage> d;
        for (int s = 0; s < cfg.layers + 2; s++)
        {
            GrayImage difference(g[s].width, g[s].height);
            for (size_t i = 0; i < difference.pixels.size(); i++)
                difference.pixels[i] = g[s + 1].pixels[i] - g[s].pixels[i];
            d.push_back(std::move(difference));
        }
        for (int s = 1; s <= cfg.layers; s++)
            for (int y = 5; y < d[s].height - 5; y++)
                for (int x = 5; x < d[s].width - 5; x++)
                {
                    double v = d[s].at(x, y);
                    if (std::abs(v) < .5 * cfg.contrast / cfg.layers)
                        continue;
                    bool extremum = true;
                    for (int ds = -1; ds <= 1 && extremum; ds++)
                        for (int dy = -1; dy <= 1 && extremum; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (!dx && !dy && !ds)
                                    continue;
                                double other = d[s + ds].at(x + dx, y + dy);
                                if (v > 0 ? v <= other : v >= other)
                                {
                                    extremum = false;
                                    break;
                                }
                            }
                    if (!extremum)
                        continue;
                    Candidate p;
                    p.octave = octave;
                    if (!localize(d, x, y, s, cfg, p))
                        continue;
                    for (auto angle : orientations(g[p.layer], p))
                    {
                        p.angle = angle;
                        candidates.push_back(p);
                    }
                }
        if (std::min(g[cfg.layers].width, g[cfg.layers].height) >= 32)
            base = halfSample(g[cfg.layers]);
        else
            base = GrayImage{};
        gaussians.push_back(std::move(g));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        if (a.response != b.response)
            return a.response > b.response;
        if (a.octave != b.octave)
            return a.octave < b.octave;
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        if (a.sigma != b.sigma)
            return a.sigma < b.sigma;
        return a.angle < b.angle;
    });
    FeatureSet result;
    result.extractorId = "native-dog-sift-v1";
    result.descriptorDimension = 128;
    // Eliminate duplicate localized extrema before enforcing the requested cap.
    for (const auto &p : candidates)
    {
        double scale = std::pow(2., p.octave);
        Keypoint key{float(p.x * scale), float(p.y * scale), float(2 * p.sigma * scale), float(p.angle)};
        bool duplicate = false;
        for (const auto &q : result.keypoints)
            if (std::abs(q.x - key.x) < .01 && std::abs(q.y - key.y) < .01 &&
                std::abs(q.scale - key.scale) < .01 && std::abs(q.orientation - key.orientation) < .01)
            {
                duplicate = true;
                break;
            }
        if (duplicate)
            continue;
        auto desc = descriptor(gaussians[p.octave + 1][p.layer], p);
        double norm = 0;
        for (float v : desc)
            norm += v * v;
        if (norm < .5)
            continue;
        result.keypoints.push_back(key);
        result.descriptors.insert(result.descriptors.end(), desc.begin(), desc.end());
        if (result.keypoints.size() >= cfg.maximumFeatures)
            break;
    }
    return result;
}
void rootSift(std::vector<float> &descriptors)
{
    if (descriptors.size() % 128)
        throw std::runtime_error("Invalid RootSIFT descriptor dimension.");
    for (size_t i = 0; i < descriptors.size(); i += 128)
    {
        double sum = 0;
        for (size_t j = 0; j < 128; j++)
        {
            float v = descriptors[i + j];
            if (!std::isfinite(v) || v < 0)
                throw std::runtime_error("Invalid RootSIFT input.");
            sum += v;
        }
        if (sum <= 0)
            throw std::runtime_error("Zero RootSIFT descriptor.");
        double norm = 0;
        for (size_t j = 0; j < 128; j++)
        {
            auto &v = descriptors[i + j];
            v = float(std::sqrt(std::max(1e-6, double(v) / sum)));
            norm += double(v) * v;
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < 128; j++)
            descriptors[i + j] = float(descriptors[i + j] / norm);
    }
}
} // namespace objectkey::vision
