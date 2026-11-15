#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace objectkey {
enum class Status { Ok, InvalidInput, IoError, NotImplemented };
struct Outcome {
    Status status = Status::NotImplemented;
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
struct PipelineConfig {
    std::string version = "1";
    std::string preprocessingId = "not-implemented";
    std::string extractorId = "not-implemented";
    // Unset until validation has selected thresholds; never silently approve.
    std::optional<double> appearanceThreshold, viewpointThreshold;
};
struct ObjectTemplate {
    std::uint32_t schemaVersion = 1;
    std::string objectId;
    Roi referenceRoi;
    PipelineConfig config;
    FeatureSet features;
};
struct Match { std::size_t referenceIndex = 0, queryIndex = 0; double distance = 0; };
struct MatchResult { std::vector<Match> matches; };
struct GeometryResult {
    bool valid = false;
    std::array<double, 9> homography{};
    std::vector<std::size_t> inlierMatchIndices;
    std::optional<double> reprojectionError, viewpointDeviation;
};
enum class AuthDecision { NotEvaluated, Reject, Accept };
struct AuthResult {
    AuthDecision decision = AuthDecision::NotEvaluated;
    std::optional<double> appearanceScore, viewpointDeviation;
    std::string reason = "Authentication is not implemented.";
};
struct EvaluationSample {
    std::filesystem::path imagePath;
    bool expectedAccept = false;
};
struct EvaluationReport {
    std::size_t sampleCount = 0;
    std::vector<AuthResult> results;
};
}
