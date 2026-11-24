#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace objectkey {
struct MatchingThresholds {
    double appearance=.8;
    std::size_t minimumInliers=10;
    double minimumInlierRatio=.70;
    double ransacReprojectionRatio=.025;
    void validate() const {
        if(!std::isfinite(appearance)||appearance< -1||appearance>1)
            throw std::runtime_error("appearance_threshold must be finite and within [-1,1].");
        if(minimumInliers<3||minimumInliers>500)
            throw std::runtime_error("minimum_inliers must be an integer within [3,500].");
        if(!std::isfinite(minimumInlierRatio)||minimumInlierRatio<0||minimumInlierRatio>1)
            throw std::runtime_error("minimum_inlier_ratio must be finite and within [0,1].");
        if(!std::isfinite(ransacReprojectionRatio)||ransacReprojectionRatio<=0||ransacReprojectionRatio>1)
            throw std::runtime_error("ransac_reprojection_ratio must be finite and within (0,1].");
    }
};
}
