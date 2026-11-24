#include "vision/GrabCut.h"
#include "vision/MaxFlow.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace objectkey::vision
{
namespace
{
using Color = std::array<double, 3>;
struct Component
{
    Color mean{};
    std::array<double, 9> inverse{};
    double weight = 0, logDet = 0;
    double logProbability(const Color &color) const
    {
        if (weight <= 0)
            return -std::numeric_limits<double>::infinity();
        Color d{color[0] - mean[0], color[1] - mean[1], color[2] - mean[2]};
        double distance = 0;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                distance += d[r] * inverse[3 * r + c] * d[c];
        return std::log(weight) - .5 * (logDet + std::max(0., distance) + 3 * std::log(2 * std::numbers::pi));
    }
    void covariance(const std::array<double, 9> &covariance)
    {
        // Cholesky factorization plus triangular inversion. The diagonal floor
        // regularizes flat-color / rank-deficient components in RGB units.
        for (double jitter = .01; jitter <= 10000; jitter *= 10)
        {
            std::array<double, 9> l{};
            bool ok = true;
            for (int i = 0; i < 3 && ok; i++)
                for (int j = 0; j <= i; j++)
                {
                    double sum = covariance[3 * i + j] + (i == j ? jitter : 0);
                    for (int k = 0; k < j; k++)
                        sum -= l[3 * i + k] * l[3 * j + k];
                    if (i == j)
                    {
                        if (!std::isfinite(sum) || sum <= 1e-12)
                        {
                            ok = false;
                            break;
                        }
                        l[3 * i + j] = std::sqrt(sum);
                    }
                    else
                        l[3 * i + j] = sum / l[3 * j + j];
                }
            if (!ok)
                continue;
            std::array<double, 9> li{};
            for (int column = 0; column < 3; column++)
                for (int i = 0; i < 3; i++)
                {
                    double v = i == column ? 1. : 0.;
                    for (int k = 0; k < i; k++)
                        v -= l[3 * i + k] * li[3 * k + column];
                    li[3 * i + column] = v / l[3 * i + i];
                }
            inverse.fill(0);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    for (int k = 0; k < 3; k++)
                        inverse[3 * r + c] += li[3 * k + r] * li[3 * k + c];
            logDet = 2 * (std::log(l[0]) + std::log(l[4]) + std::log(l[8]));
            return;
        }
        throw std::runtime_error("Cannot regularize GrabCut covariance.");
    }
};
struct Gmm
{
    std::array<Component, 5> components{};
    int assign(const Color &color) const
    {
        int best = 0;
        double score = -INFINITY;
        for (int k = 0; k < 5; k++)
        {
            double value = components[k].logProbability(color);
            if (value > score)
            {
                score = value;
                best = k;
            }
        }
        return best;
    }
    double cost(const Color &color) const
    {
        std::array<double, 5> log{};
        double peak = -INFINITY;
        for (int k = 0; k < 5; k++)
            peak = std::max(peak, log[k] = components[k].logProbability(color));
        if (!std::isfinite(peak))
            throw std::runtime_error("Empty GrabCut color model.");
        double sum = 0;
        for (auto value : log)
            sum += std::exp(value - peak);
        return -peak - std::log(sum);
    }
    void fit(const std::vector<Color> &colors, const std::vector<int> &indices,
             const std::vector<int> &assignment)
    {
        if (indices.empty())
            throw std::runtime_error("GrabCut requires both foreground and background samples.");
        std::array<size_t, 5> count{};
        std::array<Color, 5> sums{};
        std::array<std::array<double, 9>, 5> products{};
        for (int index : indices)
        {
            int k = assignment[index];
            count[k]++;
            const auto &v = colors[index];
            for (int r = 0; r < 3; r++)
            {
                sums[k][r] += v[r];
                for (int c = 0; c < 3; c++)
                    products[k][3 * r + c] += v[r] * v[c];
            }
        }
        for (int k = 0; k < 5; k++)
        {
            auto &component = components[k];
            component.weight = double(count[k]) / indices.size();
            if (!count[k])
                continue;
            for (int c = 0; c < 3; c++)
                component.mean[c] = sums[k][c] / count[k];
            std::array<double, 9> covariance{};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    covariance[3 * r + c] =
                        products[k][3 * r + c] / count[k] - component.mean[r] * component.mean[c];
            component.covariance(covariance);
        }
    }
    void initialize(const std::vector<Color> &colors, const std::vector<int> &indices,
                    std::vector<int> &assignment, std::mt19937 &rng)
    {
        if (indices.empty())
            throw std::runtime_error("GrabCut requires background outside the object ROI.");
        std::array<Color, 5> means{};
        means[0] = colors[indices[std::uniform_int_distribution<size_t>(0, indices.size() - 1)(rng)]];
        auto squared = [](const Color &a, const Color &b) {
            double v = 0;
            for (int c = 0; c < 3; c++)
                v += (a[c] - b[c]) * (a[c] - b[c]);
            return v;
        };
        std::vector<double> nearest(indices.size(), INFINITY);
        for (int k = 1; k < 5; k++)
        {
            double sum = 0;
            for (size_t i = 0; i < indices.size(); i++)
                sum += (nearest[i] = std::min(nearest[i], squared(colors[indices[i]], means[k - 1])));
            double draw = std::uniform_real_distribution<double>(0, std::max(sum, 1e-30))(rng);
            size_t selected = 0;
            for (; selected + 1 < indices.size() && draw > nearest[selected]; selected++)
                draw -= nearest[selected];
            means[k] = colors[indices[selected]];
        }
        for (int iteration = 0; iteration < 12; iteration++)
        {
            size_t changed = 0;
            std::array<Color, 5> sums{};
            std::array<size_t, 5> count{};
            for (int index : indices)
            {
                int best = 0;
                double distance = INFINITY;
                for (int k = 0; k < 5; k++)
                {
                    double d = squared(colors[index], means[k]);
                    if (d < distance)
                    {
                        distance = d;
                        best = k;
                    }
                }
                changed += assignment[index] != best;
                assignment[index] = best;
                count[best]++;
                for (int c = 0; c < 3; c++)
                    sums[best][c] += colors[index][c];
            }
            for (int k = 0; k < 5; k++)
                if (count[k])
                    for (int c = 0; c < 3; c++)
                        means[k][c] = sums[k][c] / count[k];
            if (iteration > 0 && !changed)
                break;
        }
        fit(colors, indices, assignment);
    }
};
bool isForeground(unsigned char label)
{
    return label == Foreground || label == ProbableForeground;
}
} // namespace
Mask rectangleLabels(int w, int h, Rect rectangle)
{
    Mask labels(w, h, Background);
    rectangle = intersect(rectangle, {0, 0, w, h});
    if (rectangle.width < 2 || rectangle.height < 2 ||
        size_t(rectangle.width) * rectangle.height >= labels.pixels.size())
        throw std::runtime_error("GrabCut rectangle must leave some background.");
    for (int y = rectangle.y; y < rectangle.y + rectangle.height; y++)
        for (int x = rectangle.x; x < rectangle.x + rectangle.width; x++)
            labels.at(x, y) = ProbableForeground;
    return labels;
}
Mask foregroundMask(const Mask &labels)
{
    Mask result(labels.width, labels.height);
    for (size_t i = 0; i < labels.pixels.size(); i++)
        result.pixels[i] = isForeground(labels.pixels[i]) ? 1 : 0;
    return result;
}
Mask grabCut(const Image &image, Mask labels, unsigned iterations)
{
    validateImage(image);
    const int w = int(image.width), h = int(image.height), n = w * h;
    if (n > 2000000 || labels.width != w || labels.height != h || labels.pixels.size() != size_t(n) ||
        !iterations || iterations > 20)
        throw std::runtime_error("Invalid GrabCut canvas or iteration count.");
    std::vector<Color> colors(n);
    std::vector<int> background, foreground, assignment(n, -1);
    for (int i = 0; i < n; i++)
    {
        if (labels.pixels[i] > 3)
            throw std::runtime_error("Invalid GrabCut label.");
        for (int c = 0; c < 3; c++)
            colors[i][c] = image.pixels[size_t(i) * 3 + c];
        (isForeground(labels.pixels[i]) ? foreground : background).push_back(i);
    }
    std::mt19937 random(20261119);
    Gmm bg, fg;
    bg.initialize(colors, background, assignment, random);
    fg.initialize(colors, foreground, assignment, random);
    struct Neighbor
    {
        int a, b;
        double distance, weight;
    };
    std::vector<Neighbor> neighbors;
    neighbors.reserve(size_t(n) * 4);
    double squaredSum = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (auto delta : std::array<std::array<int, 2>, 4>{{{-1, 0}, {0, -1}, {-1, -1}, {1, -1}}})
            {
                int xx = x + delta[0], yy = y + delta[1];
                if (xx < 0 || xx >= w || yy < 0)
                    continue;
                int a = y * w + x, b = yy * w + xx;
                double d = 0;
                for (int c = 0; c < 3; c++)
                    d += std::pow(colors[a][c] - colors[b][c], 2);
                squaredSum += d;
                neighbors.push_back({a, b, d, delta[0] && delta[1] ? 50 / std::sqrt(2.) : 50.});
            }
    double beta = squaredSum > 1e-12 ? neighbors.size() / (2 * squaredSum) : 0;
    for (auto &edge : neighbors)
        edge.weight *= std::exp(-beta * edge.distance);
    for (unsigned iteration = 0; iteration < iterations; iteration++)
    {
        background.clear();
        foreground.clear();
        for (int i = 0; i < n; i++)
        {
            bool f = isForeground(labels.pixels[i]);
            assignment[i] = (f ? fg : bg).assign(colors[i]);
            (f ? foreground : background).push_back(i);
        }
        bg.fit(colors, background, assignment);
        fg.fit(colors, foreground, assignment);
        FlowGraph graph(n + 2);
        graph.reserve(neighbors.size() * 2 + size_t(n) * 4);
        const int source = n, sink = n + 1;
        for (int i = 0; i < n; i++)
        {
            double dBackground = 0, dForeground = 0;
            auto label = labels.pixels[i];
            if (label == Background)
                dForeground = 450;
            else if (label == Foreground)
                dBackground = 450;
            else
            {
                dBackground = bg.cost(colors[i]);
                dForeground = fg.cost(colors[i]);
                double offset = std::min(dBackground, dForeground);
                dBackground -= offset;
                dForeground -= offset;
            }
            // Source side is foreground: cutting i->sink pays its foreground cost.
            graph.addEdge(source, i, dBackground);
            graph.addEdge(i, sink, dForeground);
        }
        for (const auto &edge : neighbors)
            graph.addEdge(edge.a, edge.b, edge.weight, edge.weight);
        graph.maxFlow(source, sink);
        auto reached = graph.sourceSide(source);
        for (int i = 0; i < n; i++)
            if (labels.pixels[i] >= 2)
                labels.pixels[i] = reached[i] ? ProbableForeground : ProbableBackground;
        if (!foregroundCount(foregroundMask(labels)))
            throw std::runtime_error(
                "No usable GrabCut foreground. Adjust the ROI or paint foreground strokes.");
    }
    return labels;
}
} // namespace objectkey::vision
