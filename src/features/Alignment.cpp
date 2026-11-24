#include "features/Alignment.h"
#include "vision/Affine.h"
#include "vision/GrabCut.h"
#include "vision/Sift.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>
namespace objectkey
{
namespace
{
vision::Mask maskOf(const LocalFeatures &f)
{
    vision::Mask mask(int(f.width), int(f.height));
    mask.pixels = f.objectMask;
    return mask;
}
} // namespace
void validateSpatialFeatures(const LocalFeatures &f)
{
    if (f.features.extractorId != spatialExtractorId || f.features.descriptorDimension != 128 || !f.width ||
        !f.height || f.width > 320 || f.height > 320 || f.objectMask.size() != size_t(f.width) * f.height ||
        f.features.keypoints.size() > 500 ||
        f.features.descriptors.size() != f.features.keypoints.size() * 128)
        throw std::runtime_error(
            "Invalid spatial template metadata. Re-extract templates made by the previous OpenCV pipeline.");
    if (f.regionSource != "supplied-mask-v1" && f.regionSource != "grabcut-native5-v1")
        throw std::runtime_error("Unsupported object mask source.");
    if (!f.chroma.empty() && f.chroma.size() != size_t(f.width) * f.height * 2)
        throw std::runtime_error("Invalid color context dimensions.");
    for (float v : f.chroma)
        if (!std::isfinite(v) || v < -128 || v > 127)
            throw std::runtime_error("Invalid color context value.");
    if (!f.gray.empty() && f.gray.size() != size_t(f.width) * f.height)
        throw std::runtime_error("Invalid luminance context dimensions.");
    for (float v : f.gray)
        if (!std::isfinite(v) || v < 0 || v > 255.001f)
            throw std::runtime_error("Invalid luminance context value.");
    for (auto v : f.objectMask)
        if (v > 1)
            throw std::runtime_error("Invalid binary object mask.");
    auto mask = maskOf(f);
    auto box = vision::bounds(mask);
    if (vision::foregroundCount(mask) < 64 || box.width < 8 || box.height < 8)
        throw std::runtime_error("Object mask is empty or too small.");
    for (auto p : f.features.keypoints)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.scale) ||
            !std::isfinite(p.orientation) || p.x < 0 || p.y < 0 || p.x >= f.width || p.y >= f.height ||
            p.scale <= 0 || !f.objectMask[size_t(p.y) * f.width + size_t(p.x)])
            throw std::runtime_error("Invalid spatial keypoint.");
    for (size_t i = 0; i < f.features.keypoints.size(); i++)
    {
        double norm = 0;
        for (size_t j = 0; j < 128; j++)
        {
            double v = f.features.descriptors[i * 128 + j];
            if (!std::isfinite(v) || v < 0 || v > 1)
                throw std::runtime_error("Invalid local descriptor.");
            norm += v * v;
        }
        if (std::abs(norm - 1) > 1e-4)
            throw std::runtime_error("Invalid descriptor norm.");
    }
}
Image makeGrabCutMask(const Image &image, const Roi &roi)
{
    vision::validateRoi(image, roi);
    double scale = std::min({1., 768. / image.width, 570. / image.height});
    auto working = vision::resizeArea(image, std::max(1, int(std::lround(image.width * scale))),
                                      std::max(1, int(std::lround(image.height * scale))));
    if (working.width < 10 || working.height < 10)
        throw std::runtime_error("Image is too small for GrabCut.");
    vision::Rect box{int(std::lround(roi.x * scale)), int(std::lround(roi.y * scale)),
                     int(std::lround(roi.width * scale)), int(std::lround(roi.height * scale))};
    box = vision::intersect(box, {1, 1, int(working.width) - 2, int(working.height) - 2});
    auto labels =
        vision::grabCut(working, vision::rectangleLabels(int(working.width), int(working.height), box));
    auto small = vision::foregroundMask(labels);
    if (vision::foregroundCount(small) < 64)
        throw std::runtime_error(
            "No usable GrabCut foreground. Adjust the object ROI or use the mask editor.");
    return vision::maskImage(vision::resizeNearest(small, int(image.width), int(image.height)));
}
LocalFeatures extractLocal(const Image &image, const Roi &roi, const Image *suppliedMask)
{
    vision::validateRoi(image, roi);
    if (!suppliedMask)
        throw std::runtime_error("A foreground mask is required for local extraction.");
    if (suppliedMask->width != image.width || suppliedMask->height != image.height)
        throw std::runtime_error("Object mask dimensions must match the input image.");
    double scale = std::min(1., 320. / std::max(roi.width, roi.height));
    int w = std::max(1, int(std::lround(roi.width * scale))),
        h = std::max(1, int(std::lround(roi.height * scale)));
    auto canvas = vision::resizeArea(
        vision::crop(image, {int(roi.x), int(roi.y), int(roi.width), int(roi.height)}), w, h);
    auto mask = vision::resizeNearest(vision::readMask(*suppliedMask, roi), w, h);
    auto box = vision::bounds(mask);
    if (vision::foregroundCount(mask) < 64 || box.width < 8 || box.height < 8)
        throw std::runtime_error("Object mask is empty or too small.");
    LocalFeatures out;
    out.width = unsigned(w);
    out.height = unsigned(h);
    out.objectMask = mask.pixels;
    out.regionSource = "supplied-mask-v1";
    out.features.extractorId = spatialExtractorId;
    out.features.descriptorDimension = 128;
    auto gray = vision::grayscale(canvas);
    auto working = vision::resizeBilinear(gray, w * 2, h * 2);
    for (auto &v : working.pixels)
        v = float(std::round(v * 255) / 255.);
    auto features = vision::sift(working);
    std::set<std::tuple<long, long, long>> seen;
    for (size_t i = 0; i < features.keypoints.size(); i++)
    {
        auto p = features.keypoints[i];
        auto id = std::make_tuple(long(std::nearbyint(p.x * 100)), long(std::nearbyint(p.y * 100)),
                                  long(std::nearbyint(p.scale * 100)));
        if (!seen.insert(id).second)
            continue;
        float x = (p.x + .5f) / 2, y = (p.y + .5f) / 2;
        if (x < 0 || y < 0 || x >= w || y >= h || !mask.at(int(x), int(y)))
            continue;
        out.features.keypoints.push_back({x, y, p.scale / 2, p.orientation});
        out.features.descriptors.insert(out.features.descriptors.end(),
                                        features.descriptors.begin() + i * 128,
                                        features.descriptors.begin() + (i + 1) * 128);
    }
    // The V5 storage layout retains these fields; neither participates in matching.
    out.chroma = vision::labChroma(canvas);
    auto blurred = vision::gaussian(gray, .8, 1);
    out.gray = std::move(blurred.pixels);
    for (auto &value : out.gray)
        value *= 255;
    validateSpatialFeatures(out);
    return out;
}
AlignmentResult checkObjectMatches(const LocalFeatures &ref, const LocalFeatures &query,
                                   const FeatureMatches &pairs, const MatchingThresholds &thresholds)
{
    thresholds.validate();
    validateSpatialFeatures(ref);
    validateSpatialFeatures(query);
    AlignmentResult result;
    const auto &r = ref.features;
    const auto &q = query.features;
    if (r.keypoints.size() < 3 || q.keypoints.size() < 3)
    {
        result.reason = "INSUFFICIENT_FEATURES";
        return result;
    }
    std::vector<vision::Point> from, to;
    std::set<size_t> usedR, usedQ;
    for (auto [a, b] : pairs)
    {
        if (a >= r.keypoints.size() || b >= q.keypoints.size() || !usedR.insert(a).second ||
            !usedQ.insert(b).second)
            throw std::runtime_error("Invalid or duplicate feature correspondence.");
        from.push_back({r.keypoints[a].x, r.keypoints[a].y});
        to.push_back({q.keypoints[b].x, q.keypoints[b].y});
    }
    result.matches = from.size();
    if (from.size() < 3)
    {
        result.reason = "INSUFFICIENT_MATCHES";
        return result;
    }
    auto box = vision::bounds(maskOf(query));
    double tolerance = thresholds.ransacReprojectionRatio * std::hypot(box.width, box.height);
    auto fit = vision::ransacAffine(from, to, tolerance);
    result.reason = "GEOMETRY";
    if (!fit)
        return result;
    result.inliers = fit->inliers.size();
    result.ratio = double(result.inliers) / result.matches;
    result.reason = result.inliers < thresholds.minimumInliers     ? "INSUFFICIENT_INLIERS"
                    : result.ratio < thresholds.minimumInlierRatio ? "INLIER_RATIO"
                                                                   : "ALIGNED";
    result.accepted = result.reason == "ALIGNED";
    return result;
}
} // namespace objectkey
