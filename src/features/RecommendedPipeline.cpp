#include "features/RecommendedPipeline.h"
#include "vision/Affine.h"
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
namespace objectkey
{
namespace
{
std::filesystem::path matcherPath(const std::filesystem::path &model,
                                  const std::filesystem::path &explicitPath)
{
    if (!explicitPath.empty())
        return explicitPath;
    auto adjacent = model.parent_path() / "lightglue_sift";
    if (std::filesystem::exists(adjacent / "manifest.txt"))
        return adjacent;
    wchar_t buffer[32768]{};
    GetModuleFileNameW(nullptr, buffer, 32768);
    auto directory = std::filesystem::path(buffer).parent_path();
    for (int i = 0; i < 6 && !directory.empty(); i++)
    {
        auto candidate = directory / "models/lightglue_sift";
        if (std::filesystem::exists(candidate / "manifest.txt"))
            return candidate;
        directory = directory.parent_path();
    }
    return "models/lightglue_sift";
}
} // namespace
RecommendedPipeline::RecommendedPipeline(const std::filesystem::path &resnet,
                                         const std::filesystem::path &directory,
                                         const MatchingThresholds &thresholds)
    : thresholds_(thresholds), resnet_(resnet), matcher_(matcherPath(resnet, directory))
{
    thresholds_.validate();
}
void RecommendedPipeline::requireCurrentTemplate(const EmbeddingTemplate &record)
{
    if (record.preprocessing != canonicalPreprocessId || record.matcherSha256.empty() ||
        record.local.gray.empty())
        throw std::runtime_error("Re-extract the original image using the native pipeline; previous OpenCV "
                                 "templates are incompatible.");
    validateSpatialFeatures(record.local);
}
void RecommendedPipeline::validateTemplate(const EmbeddingTemplate &record) const
{
    requireCurrentTemplate(record);
    if (record.modelSha256 != resnet_.modelHash() || record.matcherSha256 != matcher_.fingerprint())
        throw std::runtime_error(
            "Template model fingerprint differs. Use the original models or re-register.");
}
EmbeddingTemplate RecommendedPipeline::extract(const Image &image, const Roi &roi, const Image *supplied)
{
    vision::validateRoi(image, roi);
    const bool automatic = supplied == nullptr;
    Image generated;
    if (automatic)
    {
        generated = makeGrabCutMask(image, roi);
        supplied = &generated;
    }
    if (supplied->width != image.width || supplied->height != image.height)
        throw std::runtime_error("Object mask dimensions must match the original image.");
    auto mask = vision::readMask(*supplied, roi);
    if (vision::foregroundCount(mask) < 64)
        throw std::runtime_error("Object mask is empty or too small. Correct the mask and try again.");
    auto box = vision::bounds(mask);
    int dx = int(std::ceil(.05 * box.width)), dy = int(std::ceil(.05 * box.height));
    auto padded = vision::intersect({box.x - dx, box.y - dy, box.width + 2 * dx, box.height + 2 * dy},
                                    {0, 0, mask.width, mask.height});
    Roi tight{unsigned(roi.x + padded.x), unsigned(roi.y + padded.y), unsigned(padded.width),
              unsigned(padded.height)};
    auto bgr = vision::crop(image, {int(tight.x), int(tight.y), int(tight.width), int(tight.height)});
    mask = vision::crop(mask, padded);
    auto moments = vision::moments(mask);
    double theta = .5 * std::atan2(2 * moments.xy, moments.xx - moments.yy) - std::numbers::pi / 2;
    double co = std::cos(theta), si = std::sin(theta);
    vision::AffineMatrix transform{co,  si, (1 - co) * moments.cx - si * moments.cy,
                                   -si, co, si * moments.cx + (1 - co) * moments.cy};
    double minx = INFINITY, miny = INFINITY, maxx = -INFINITY, maxy = -INFINITY;
    for (int y = 0; y < mask.height; y++)
        for (int x = 0; x < mask.width; x++)
            if (mask.at(x, y))
            {
                auto p = vision::transform({double(x), double(y)}, transform);
                minx = std::min(minx, p.x);
                maxx = std::max(maxx, p.x);
                miny = std::min(miny, p.y);
                maxy = std::max(maxy, p.y);
            }
    double scale = 224. / (1.1 * std::max(maxx - minx + 1, maxy - miny + 1));
    for (auto &v : transform)
        v *= scale;
    transform[2] += 111.5 - scale * (minx + maxx) / 2;
    transform[5] += 111.5 - scale * (miny + maxy) / 2;
    for (size_t i = 0; i < mask.pixels.size(); i++)
        if (!mask.pixels[i])
        {
            bgr.pixels[i * 3] = 104;
            bgr.pixels[i * 3 + 1] = 116;
            bgr.pixels[i * 3 + 2] = 124;
        }
    auto canonical = vision::warpAffine(bgr, transform, 224, 224);
    auto first = resnet_.extract(canonical, {0, 0, 224, 224});
    auto second = resnet_.extract(vision::rotate180(canonical), {0, 0, 224, 224});
    for (size_t i = 0; i < first.size(); i++)
        first[i] += second[i];
    normalizeEmbedding(first);
    EmbeddingTemplate record{resnet_.modelHash(), tight, first, extractLocal(image, tight, supplied)};
    if (automatic)
        record.local.regionSource = "grabcut-native5-v1";
    record.preprocessing = canonicalPreprocessId;
    record.matcherSha256 = matcher_.fingerprint();
    return record;
}
AlignmentResult RecommendedPipeline::compare(const EmbeddingTemplate &reference,
                                             const EmbeddingTemplate &query)
{
    validateTemplate(reference);
    validateTemplate(query);
    unsigned layers = 0;
    auto matches = matcher_.match(reference.local, query.local, &layers);
    auto result = checkObjectMatches(reference.local, query.local, matches, thresholds_);
    result.matcherLayers = layers;
    return result;
}
} // namespace objectkey
