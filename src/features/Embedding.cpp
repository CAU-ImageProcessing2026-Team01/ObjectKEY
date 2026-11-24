#include "features/ResNet18.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace objectkey {
std::vector<float> preprocessResNet(const Image& image, const Roi& roi) {
    if (!roi.width || !roi.height || roi.x >= image.width || roi.y >= image.height ||
        roi.width > image.width - roi.x || roi.height > image.height - roi.y ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height * 3)
        throw std::runtime_error("Invalid ROI or image buffer.");
    constexpr int size = 224;
    constexpr float mean[] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[] = {0.229f, 0.224f, 0.225f};
    const double scale = std::min(double(size) / roi.width, double(size) / roi.height);
    const int width = std::clamp(int(std::lround(roi.width * scale)), 1, size);
    const int height = std::clamp(int(std::lround(roi.height * scale)), 1, size);
    const int left = (size - width) / 2, top = (size - height) / 2;
    std::vector<float> tensor(3 * size * size, 0.0f); // mean-color padding normalizes to zero
    for (int y = 0; y < height; ++y) {
        const double sy = std::clamp((y + 0.5) * roi.height / height - 0.5, 0.0, double(roi.height - 1));
        const auto y0 = static_cast<std::uint32_t>(sy), y1 = std::min(y0 + 1, roi.height - 1);
        const double fy = sy - y0;
        for (int x = 0; x < width; ++x) {
            const double sx = std::clamp((x + 0.5) * roi.width / width - 0.5, 0.0, double(roi.width - 1));
            const auto x0 = static_cast<std::uint32_t>(sx), x1 = std::min(x0 + 1, roi.width - 1);
            const double fx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                auto pixel = [&](std::uint32_t px, std::uint32_t py) {
                    return double(image.pixels[(static_cast<std::size_t>(roi.y + py) * image.width + roi.x + px) * 3 + (2 - c)]);
                };
                const double value = (1-fy)*((1-fx)*pixel(x0,y0)+fx*pixel(x1,y0)) + fy*((1-fx)*pixel(x0,y1)+fx*pixel(x1,y1));
                tensor[static_cast<std::size_t>(c) * size * size + (top+y)*size + left+x] =
                    static_cast<float>((value / 255.0 - mean[c]) / deviation[c]);
            }
        }
    }
    return tensor;
}
void normalizeEmbedding(Embedding& embedding) {
    double sum = 0;
    for (float value : embedding) {
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite embedding.");
        sum += double(value) * value;
    }
    if (sum < 1e-24) throw std::runtime_error("Zero-length embedding.");
    const double norm = std::sqrt(sum);
    for (auto& value : embedding) value = static_cast<float>(value / norm);
}
namespace {
void validateUnit(const Embedding& embedding) {
    double norm2 = 0;
    for (float x : embedding) {
        if (!std::isfinite(x)) throw std::runtime_error("Non-finite template embedding.");
        norm2 += double(x)*x;
    }
    if (std::abs(norm2-1.0) > 1e-4) throw std::runtime_error("Embedding is not L2 normalized.");
}
}
double cosineSimilarity(const Embedding& a, const Embedding& b) {
    validateUnit(a); validateUnit(b);
    double dot = 0;
    for (std::size_t i=0; i<a.size(); ++i) dot += double(a[i])*b[i];
    return std::clamp(dot, -1.0, 1.0);
}
void saveEmbedding(const std::filesystem::path& path, const EmbeddingTemplate& record) {
    validateUnit(record.embedding);
    if(record.local.width) validateSpatialFeatures(record.local);
    if(record.preprocessing!=canonicalPreprocessId)throw std::runtime_error("Unsupported preprocessing. Re-register the original image.");
    if((record.matcherSha256.size()!=64||record.matcherSha256.find_first_not_of("0123456789abcdef")!=std::string::npos||record.local.gray.size()!=size_t(record.local.width)*record.local.height||!record.local.width||record.local.chroma.empty()))
        throw std::runtime_error("Incomplete current-pipeline template.");
    if (std::filesystem::exists(path)) throw std::runtime_error("Template already exists; choose a new output path.");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.imbue(std::locale::classic());
    if (!out) throw std::runtime_error("Cannot create template.");
    out << "OBJECTKEY_EMBEDDING_V5\nresnet18-imagenet1k-v1-fc-identity\n" << record.preprocessing << '\n' << record.modelSha256 << '\n';
    out << record.matcherSha256 << '\n';
    out << record.roi.x << ' ' << record.roi.y << ' ' << record.roi.width << ' ' << record.roi.height << "\n512\n";
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : record.embedding) out << value << '\n';
    if(record.local.width) {
        const auto& f=record.local.features;
        out << f.extractorId << ' ' << record.local.width << ' ' << record.local.height << ' ' << f.keypoints.size() << ' ' << f.descriptorDimension << '\n';
        for(const auto& p:f.keypoints) out << p.x << ' ' << p.y << ' ' << p.scale << ' ' << p.orientation << '\n';
        for(float v:f.descriptors) out << v << '\n';
        out << recommendedPolicyId << ' ' << record.local.regionSource << ' ' << record.local.objectMask.size() << '\n';
        for(auto v:record.local.objectMask) out << (v?'1':'0');
        out << '\n';
        if(!record.local.chroma.empty()){out << "lab-ab-canvas-v1 " << record.local.chroma.size() << '\n';for(float v:record.local.chroma)out << v << '\n';}
        {out << "gray-blur08-canvas-v1 " << record.local.gray.size() << '\n';for(float v:record.local.gray)out << v << '\n';}
    }
    out.close();
    if (!out) throw std::runtime_error("Failed to finish writing template.");
}
EmbeddingTemplate loadEmbedding(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > 8388608) throw std::runtime_error("Missing or oversized template.");
    std::ifstream in(path, std::ios::binary); in.imbue(std::locale::classic());
    std::string magic, model, preprocess;
    auto readLine = [&](std::string& line) { std::getline(in, line); if (!line.empty() && line.back() == '\r') line.pop_back(); };
    readLine(magic); readLine(model); readLine(preprocess);
    if(magic!="OBJECTKEY_EMBEDDING_V5")throw std::runtime_error("Unsupported template version. Re-register the original image to create a V5 template.");
    if(model!="resnet18-imagenet1k-v1-fc-identity"||preprocess!=canonicalPreprocessId)throw std::runtime_error("Unsupported model or preprocessing. Re-register the original image with the native pipeline.");
    EmbeddingTemplate record;
    record.preprocessing=preprocess;
    readLine(record.modelSha256);
    {readLine(record.matcherSha256);if(record.matcherSha256.size()!=64||record.matcherSha256.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("Invalid matcher fingerprint.");}
    if (record.modelSha256.size()!=64 || record.modelSha256.find_first_not_of("0123456789abcdef")!=std::string::npos)
        throw std::runtime_error("Invalid model fingerprint.");
    std::size_t dimension=0;
    if (!(in >> record.roi.x >> record.roi.y >> record.roi.width >> record.roi.height >> dimension) || dimension!=512 || !record.roi.width || !record.roi.height)
        throw std::runtime_error("Invalid template metadata.");
    for (auto& x : record.embedding) if (!(in >> x)) throw std::runtime_error("Truncated or invalid embedding.");
    validateUnit(record.embedding);
    {
        auto& f=record.local.features;size_t count=0;
        if(!(in >> f.extractorId >> record.local.width >> record.local.height >> count >> f.descriptorDimension) ||
           f.extractorId!=spatialExtractorId || count>500 || f.descriptorDimension!=128 ||
           !record.local.width || !record.local.height || (record.local.width>320 || record.local.height>320))
            throw std::runtime_error("Invalid local feature metadata.");
        f.keypoints.resize(count);f.descriptors.resize(count*128);
        for(auto& p:f.keypoints) if(!(in>>p.x>>p.y>>p.scale>>p.orientation)||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.scale)||!std::isfinite(p.orientation)||p.x<0||p.y<0||p.x>=record.local.width||p.y>=record.local.height||p.scale<=0)
            throw std::runtime_error("Invalid keypoint.");
        for(auto& v:f.descriptors) if(!(in>>v)||!std::isfinite(v)||v<0||v>1) throw std::runtime_error("Invalid local descriptor.");
        for(size_t i=0;i<count;i++){double norm=0;for(size_t j=0;j<128;j++)norm+=double(f.descriptors[i*128+j])*f.descriptors[i*128+j];if(std::abs(norm-1)>1e-4)throw std::runtime_error("Invalid descriptor norm.");}
    }
    {
        std::string policy,mask;size_t length=0;
        // All V5 policies store identical features. The application applies its
        // current decision policy, so existing V5 registrations remain usable.
        if(!(in>>policy>>record.local.regionSource>>length) || (policy!=recommendedPolicyId&&policy!=silhouettePolicyId&&policy!=legacyRecommendedPolicyId) || length!=size_t(record.local.width)*record.local.height || !(in>>mask) || mask.size()!=length || mask.find_first_not_of("01")!=std::string::npos)
            throw std::runtime_error("Invalid spatial policy or object mask.");
        for(char v:mask)record.local.objectMask.push_back(v=='1'?1:0);
        validateSpatialFeatures(record.local);
    }
    {
        std::string id;size_t count=0;
        if(!(in>>id>>count)||id!="lab-ab-canvas-v1"||count!=size_t(record.local.width)*record.local.height*2)throw std::runtime_error("Invalid color context header.");
        record.local.chroma.resize(count);for(float& v:record.local.chroma)if(!(in>>v))throw std::runtime_error("Invalid or truncated color context.");
        validateSpatialFeatures(record.local);
    }
    {
        std::string id;size_t count=0;
        if(!(in>>id>>count)||id!="gray-blur08-canvas-v1"||count!=size_t(record.local.width)*record.local.height)throw std::runtime_error("Invalid luminance context header.");
        record.local.gray.resize(count);for(float& value:record.local.gray)if(!(in>>value))throw std::runtime_error("Truncated luminance context.");
        validateSpatialFeatures(record.local);
    }
    std::string extra;
    if (in >> extra) throw std::runtime_error("Unexpected data after embedding.");
    validateUnit(record.embedding);
    return record;
}
}
