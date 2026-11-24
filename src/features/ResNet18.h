#pragma once
#include "core/Types.h"
#include "features/Alignment.h"
#include <array>
#include <memory>

namespace objectkey {
inline constexpr const char* canonicalPreprocessId="mask-pca224-neutral-pool180-native-v1";
inline constexpr const char* recommendedPolicyId="native-sift-lightglue-inliers-v1";
inline constexpr const char* silhouettePolicyId="sift-lightglue-inliers-silhouette-v1";
inline constexpr const char* legacyRecommendedPolicyId="sift-lightglue-multishape-ncc08-v1";
using Embedding = std::array<float, 512>;
struct EmbeddingTemplate {
    std::string modelSha256;
    Roi roi;
    Embedding embedding{};
    LocalFeatures local;
    std::string preprocessing=canonicalPreprocessId;
    std::string matcherSha256;
};
std::string fileSha256(const std::filesystem::path&);
// RGB CHW float, 224x224, aspect-preserving bilinear resize + centered mean-color padding.
std::vector<float> preprocessResNet(const Image&, const Roi&);
void normalizeEmbedding(Embedding&);
double cosineSimilarity(const Embedding&, const Embedding&);
void saveEmbedding(const std::filesystem::path&, const EmbeddingTemplate&);
EmbeddingTemplate loadEmbedding(const std::filesystem::path&);
class ResNet18 {
public:
    explicit ResNet18(const std::filesystem::path& model);
    ~ResNet18();
    Embedding extract(const Image&, const Roi&);
    const std::string& modelHash() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
