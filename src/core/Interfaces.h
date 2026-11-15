#pragma once
#include "core/Types.h"

namespace objectkey {
class IImageSource {
public:
    virtual ~IImageSource() = default;
    virtual Result<Image> load(const std::filesystem::path& path) const = 0;
};
class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;
    virtual Result<Image> process(const Image&, const Roi&, const PipelineConfig&) const = 0;
};
class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;
    virtual Result<FeatureSet> extract(const Image&) const = 0;
};
class IMatcher {
public:
    virtual ~IMatcher() = default;
    virtual Result<MatchResult> match(const FeatureSet& reference, const FeatureSet& query) const = 0;
};
class IGeometryVerifier {
public:
    virtual ~IGeometryVerifier() = default;
    virtual Result<GeometryResult> verify(const FeatureSet&, const FeatureSet&, const MatchResult&) const = 0;
};
class IAuthenticator {
public:
    virtual ~IAuthenticator() = default;
    virtual Result<AuthResult> authenticate(const ObjectTemplate&, const Image&, const PipelineConfig&) const = 0;
};
class ITemplateStore {
public:
    virtual ~ITemplateStore() = default;
    virtual Outcome save(const std::filesystem::path&, const ObjectTemplate&) const = 0;
    virtual Result<ObjectTemplate> load(const std::filesystem::path&) const = 0;
};
class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    virtual Result<EvaluationReport> evaluate(const ObjectTemplate&, const std::vector<EvaluationSample>&) const = 0;
};
}
