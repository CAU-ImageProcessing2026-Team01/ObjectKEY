#pragma once
#include "vision/ImageOps.h"
#include <optional>
#include <span>
namespace objectkey::vision
{
// Partial-pivoting elimination. Returns false for singular/non-finite systems.
bool solve3(std::array<double, 9> matrix, std::array<double, 3> rhs, std::array<double, 3> &result);
Point transform(Point, const AffineMatrix &);
double determinant(const AffineMatrix &);
double minimumSingularValue(const AffineMatrix &);
std::optional<AffineMatrix> fitAffine(std::span<const Point> from, std::span<const Point> to);
struct AffineFit
{
    AffineMatrix matrix{};
    std::vector<size_t> inliers;
    unsigned iterations = 0;
};
std::optional<AffineFit> ransacAffine(std::span<const Point> from, std::span<const Point> to,
                                      double tolerance, unsigned maximumIterations = 2000,
                                      double confidence = .995);
} // namespace objectkey::vision
