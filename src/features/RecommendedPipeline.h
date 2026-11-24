#pragma once
#include "features/LightGlue.h"
#include "features/ResNet18.h"
namespace objectkey {
// Same-object / similar-view matching: appearance and affine inlier match.
class RecommendedPipeline {
public:
    explicit RecommendedPipeline(const std::filesystem::path& resnet,const std::filesystem::path& matcherDirectory={},const MatchingThresholds& thresholds={});
    EmbeddingTemplate extract(const Image&,const Roi&,const Image* mask=nullptr);
    AlignmentResult compare(const EmbeddingTemplate&,const EmbeddingTemplate&);
    void validateTemplate(const EmbeddingTemplate&) const;
    static void requireCurrentTemplate(const EmbeddingTemplate&);
private:
    MatchingThresholds thresholds_;
    ResNet18 resnet_;
    LightGlue matcher_;
};
}
