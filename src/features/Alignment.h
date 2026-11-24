#pragma once
#include "core/Types.h"
#include "features/MatchingThresholds.h"
#include <string>
#include <utility>
namespace objectkey {
inline constexpr const char* spatialExtractorId="dog-sift001-native-v1";
struct LocalFeatures {
    // Coordinates are relative to the bounded foreground crop.
    unsigned width=0,height=0;
    FeatureSet features;
    std::vector<std::uint8_t> objectMask;
    // Retained for the V5 container layout; pre-native templates need re-extraction.
    std::vector<float> chroma,gray;
    std::string regionSource="supplied-mask-v1";
};
struct AlignmentResult {
    bool accepted=false;
    std::string reason="LOCAL_FEATURES_UNAVAILABLE";
    size_t matches=0,inliers=0;
    double ratio=0;
    unsigned matcherLayers=0;
};
using FeatureMatches=std::vector<std::pair<size_t,size_t>>;
Image makeGrabCutMask(const Image&,const Roi&);
LocalFeatures extractLocal(const Image&,const Roi&,const Image* objectMask);
AlignmentResult checkObjectMatches(const LocalFeatures&,const LocalFeatures&,const FeatureMatches&,const MatchingThresholds& thresholds={});
void validateSpatialFeatures(const LocalFeatures&);
}
