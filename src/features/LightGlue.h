#pragma once
#include "features/Alignment.h"
#include <memory>
namespace objectkey {
class LightGlue {
public:
    explicit LightGlue(const std::filesystem::path& directory);
    ~LightGlue();
    FeatureMatches match(const LocalFeatures&,const LocalFeatures&,unsigned* layers=nullptr);
    const std::string& fingerprint() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
