#pragma once
#include "features/MatchingThresholds.h"
#include <filesystem>
namespace objectkey {
struct MatchingConfig {
    std::filesystem::path path;
    MatchingThresholds thresholds;
};
// Explicit path, or config/matching.ini beside the executable / in an ancestor.
MatchingConfig loadMatchingConfig(const std::filesystem::path& path={});
}
