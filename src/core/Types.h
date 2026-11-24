#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace objectkey {
enum class Status { Ok, InvalidInput, IoError };
struct Outcome {
    Status status = Status::InvalidInput;
    std::string message;
    [[nodiscard]] bool ok() const { return status == Status::Ok; }
};
template<class T> struct Result {
    Outcome outcome;
    std::optional<T> value;
    [[nodiscard]] bool ok() const { return outcome.ok() && value.has_value(); }
};
// Owned, tightly packed, row-major BGR8 pixels. No external-library types cross module boundaries.
struct Image {
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> pixels;
};
struct Roi { std::uint32_t x = 0, y = 0, width = 0, height = 0; };
struct Keypoint { float x = 0, y = 0, scale = 1, orientation = 0; };
struct FeatureSet {
    std::string extractorId;
    std::uint32_t descriptorDimension = 0;
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors; // keypoint-major, count * dimension
};
}
