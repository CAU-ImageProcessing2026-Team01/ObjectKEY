#include "vision/Affine.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace objectkey::vision
{
bool solve3(std::array<double, 9> a, std::array<double, 3> b, std::array<double, 3> &x)
{
    double scale = 0;
    for (double v : a)
    {
        if (!std::isfinite(v))
            return false;
        scale = std::max(scale, std::abs(v));
    }
    for (double v : b)
        if (!std::isfinite(v))
            return false;
    if (!scale)
        return false;
    for (int c = 0; c < 3; c++)
    {
        int p = c;
        for (int r = c + 1; r < 3; r++)
            if (std::abs(a[3 * r + c]) > std::abs(a[3 * p + c]))
                p = r;
        if (std::abs(a[3 * p + c]) < scale * 1e-12)
            return false;
        if (p != c)
        {
            for (int j = c; j < 3; j++)
                std::swap(a[3 * p + j], a[3 * c + j]);
            std::swap(b[p], b[c]);
        }
        for (int r = c + 1; r < 3; r++)
        {
            double t = a[3 * r + c] / a[3 * c + c];
            for (int j = c; j < 3; j++)
                a[3 * r + j] -= t * a[3 * c + j];
            b[r] -= t * b[c];
        }
    }
    for (int r = 2; r >= 0; r--)
    {
        double value = b[r];
        for (int j = r + 1; j < 3; j++)
            value -= a[3 * r + j] * x[j];
        x[r] = value / a[3 * r + r];
        if (!std::isfinite(x[r]))
            return false;
    }
    return true;
}
Point transform(Point p, const AffineMatrix &a)
{
    return {a[0] * p.x + a[1] * p.y + a[2], a[3] * p.x + a[4] * p.y + a[5]};
}
double determinant(const AffineMatrix &a)
{
    return a[0] * a[4] - a[1] * a[3];
}
double minimumSingularValue(const AffineMatrix &a)
{
    double x = a[0] * a[0] + a[3] * a[3], y = a[1] * a[1] + a[4] * a[4], z = a[0] * a[1] + a[3] * a[4];
    double largest = .5 * (x + y + std::hypot(x - y, 2 * z));
    return largest > 0 ? std::abs(determinant(a)) / std::sqrt(largest) : 0;
}
std::optional<AffineMatrix> fitAffine(std::span<const Point> from, std::span<const Point> to)
{
    size_t n = from.size();
    if (n < 3 || to.size() != n)
        return {};
    Point a{}, b{};
    for (size_t i = 0; i < n; i++)
    {
        if (!std::isfinite(from[i].x) || !std::isfinite(from[i].y) || !std::isfinite(to[i].x) ||
            !std::isfinite(to[i].y))
            return {};
        a.x += from[i].x;
        a.y += from[i].y;
        b.x += to[i].x;
        b.y += to[i].y;
    }
    a.x /= n;
    a.y /= n;
    b.x /= n;
    b.y /= n;
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; i++)
    {
        sa += std::pow(from[i].x - a.x, 2) + std::pow(from[i].y - a.y, 2);
        sb += std::pow(to[i].x - b.x, 2) + std::pow(to[i].y - b.y, 2);
    }
    sa = std::sqrt(sa / n);
    sb = std::sqrt(sb / n);
    if (sa < 1e-12 || sb < 1e-12)
        return {};
    std::vector<std::array<double, 5>> rows(n);
    for (size_t i = 0; i < n; i++)
        rows[i] = {(from[i].x - a.x) / sa, (from[i].y - a.y) / sa, 1., (to[i].x - b.x) / sb,
                   (to[i].y - b.y) / sb};
    // Householder QR on the three design columns; apply each reflection to both RHS columns.
    for (size_t k = 0; k < 3; k++)
    {
        double norm = 0;
        for (size_t i = k; i < n; i++)
            norm = std::hypot(norm, rows[i][k]);
        if (norm < 1e-10)
            return {};
        double alpha = -std::copysign(norm, rows[k][k]);
        std::vector<double> v(n - k);
        for (size_t i = k; i < n; i++)
            v[i - k] = rows[i][k];
        v[0] -= alpha;
        double v2 = 0;
        for (auto value : v)
            v2 += value * value;
        if (v2 < 1e-24)
            return {};
        for (size_t j = k; j < 5; j++)
        {
            double dot = 0;
            for (size_t i = k; i < n; i++)
                dot += v[i - k] * rows[i][j];
            dot *= 2 / v2;
            for (size_t i = k; i < n; i++)
                rows[i][j] -= dot * v[i - k];
        }
    }
    AffineMatrix out{};
    for (int c = 0; c < 2; c++)
    {
        std::array<double, 3> x{};
        for (int i = 2; i >= 0; i--)
        {
            double t = rows[i][3 + c];
            for (int j = i + 1; j < 3; j++)
                t -= rows[i][j] * x[j];
            x[i] = t / rows[i][i];
        }
        size_t offset = size_t(c) * 3;
        out[offset] = x[0] * sb / sa;
        out[offset + 1] = x[1] * sb / sa;
        out[offset + 2] = (c ? b.y : b.x) + sb * x[2] - out[offset] * a.x - out[offset + 1] * a.y;
    }
    for (auto value : out)
        if (!std::isfinite(value))
            return {};
    return out;
}
std::optional<AffineFit> ransacAffine(std::span<const Point> from, std::span<const Point> to,
                                      double tolerance, unsigned maximumIterations, double confidence)
{
    if (from.size() != to.size() || !std::isfinite(tolerance) || tolerance <= 0 || confidence <= 0 ||
        confidence >= 1 || !maximumIterations)
        throw std::runtime_error("Invalid affine RANSAC input.");
    const size_t n = from.size();
    if (n < 3)
        return {};
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(from[i].x) || !std::isfinite(from[i].y) || !std::isfinite(to[i].x) ||
            !std::isfinite(to[i].y))
            throw std::runtime_error("Non-finite correspondence.");
    const double limit = tolerance * tolerance;
    auto select = [&](const AffineMatrix &m, double &loss) {
        std::vector<size_t> ids;
        loss = 0.;
        for (size_t i = 0; i < n; i++)
        {
            auto p = transform(from[i], m);
            double error = std::pow(p.x - to[i].x, 2) + std::pow(p.y - to[i].y, 2);
            loss += std::min(error, limit);
            if (error <= limit)
                ids.push_back(i);
        }
        return ids;
    };
    std::mt19937 random(20261119);
    std::uniform_int_distribution<size_t> sample(0, n - 1);
    std::optional<AffineFit> best;
    double bestLoss = INFINITY;
    unsigned budget = maximumIterations, performed = 0;
    for (unsigned iteration = 0; iteration < budget; iteration++)
    {
        performed++;
        std::array<size_t, 3> index{sample(random), 0, 0};
        do
        {
            index[1] = sample(random);
        } while (index[1] == index[0]);
        do
        {
            index[2] = sample(random);
        } while (index[2] == index[0] || index[2] == index[1]);
        std::array<Point, 3> a{from[index[0]], from[index[1]], from[index[2]]},
            b{to[index[0]], to[index[1]], to[index[2]]};
        auto triangle = [](const auto &p) {
            double cross = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x);
            double scale = 0;
            for (int i = 1; i < 3; i++)
                scale = std::max(scale, std::pow(p[i].x - p[0].x, 2) + std::pow(p[i].y - p[0].y, 2));
            return std::abs(cross) > 1e-6 * std::max(1., scale);
        };
        if (!triangle(a) || !triangle(b))
            continue;
        auto model = fitAffine(a, b);
        if (!model || determinant(*model) <= 0 || minimumSingularValue(*model) < 1e-6)
            continue;
        double loss = 0;
        auto ids = select(*model, loss);
        if (ids.size() < 3 || (best && (ids.size() < best->inliers.size() ||
                                        (ids.size() == best->inliers.size() && loss >= bestLoss))))
            continue;
        best = AffineFit{*model, std::move(ids), 0};
        bestLoss = loss;
        double success = std::pow(double(best->inliers.size()) / n, 3);
        if (success >= 1)
            budget = iteration + 1;
        else if (success > 0)
        {
            double needed = std::ceil(std::log1p(-confidence) / std::log1p(-success));
            budget = std::min(budget,
                              std::max(iteration + 1, unsigned(std::min(needed, double(maximumIterations)))));
        }
    }
    if (!best)
        return {};
    best->iterations = performed;
    for (int iteration = 0; iteration < 3; iteration++)
    {
        std::vector<Point> a, b;
        for (auto i : best->inliers)
        {
            a.push_back(from[i]);
            b.push_back(to[i]);
        }
        auto model = fitAffine(a, b);
        if (!model || determinant(*model) <= 0 || minimumSingularValue(*model) < 1e-6)
            break;
        double loss = 0;
        auto ids = select(*model, loss);
        if (ids.size() < 3)
            break;
        bool stable = ids == best->inliers;
        best->matrix = *model;
        best->inliers = std::move(ids);
        if (stable)
            break;
    }
    return best;
}
} // namespace objectkey::vision
