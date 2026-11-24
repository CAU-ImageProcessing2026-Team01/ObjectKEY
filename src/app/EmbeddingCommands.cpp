#include "app/EmbeddingCommands.h"
#include "app/MatchingConfig.h"
#include "features/RecommendedPipeline.h"
#include <charconv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace objectkey {
namespace {
using Options=std::map<std::wstring,std::wstring>;
const std::wstring& required(const Options& options,const wchar_t* key) {
    auto found=options.find(key);
    if(found==options.end() || found->second.empty()) throw std::runtime_error("Missing required option; see --help.");
    return found->second;
}
Roi parseRoi(const std::wstring& value,const Image& image) {
    if(value==L"full") return {0,0,image.width,image.height};
    std::array<std::uint32_t,4> parts{};
    std::size_t begin=0;
    for(std::size_t i=0;i<4;++i) {
        const auto end=value.find(L',',begin);
        const auto part=value.substr(begin,end==std::wstring::npos?end:end-begin);
        if(part.empty() || part.find_first_not_of(L"0123456789")!=std::wstring::npos || (i<3 && end==std::wstring::npos) || (i==3 && end!=std::wstring::npos))
            throw std::runtime_error("ROI must be x,y,width,height or full.");
        const auto number=std::stoull(part);
        if(number>UINT32_MAX) throw std::runtime_error("ROI coordinate too large.");
        parts[i]=static_cast<std::uint32_t>(number); begin=end+1;
    }
    return {parts[0],parts[1],parts[2],parts[3]};
}
Image load(const IImageSource& source,const std::wstring& name) {
    auto result=source.load(std::filesystem::path(name));
    if(!result.ok()) throw std::runtime_error(result.outcome.message);
    return std::move(*result.value);
}
}
int runEmbeddingCommand(int argc,wchar_t* argv[],const IImageSource& images) {
    const std::wstring command=argv[1];
    std::set<std::wstring> allowed{L"--model",L"--matcher-dir",L"--config"};
    if(command==L"enroll") allowed.insert({L"--image",L"--roi",L"--out",L"--mask"});
    else if(command==L"verify") allowed.insert({L"--image",L"--roi",L"--template",L"--threshold",L"--mask"});
    else allowed.insert({L"--reference",L"--reference-roi",L"--query",L"--query-roi",L"--threshold",L"--reference-mask",L"--query-mask"});
    if((argc-2)%2!=0) throw std::runtime_error("Each option requires a value; see --help.");
    Options options;
    for(int i=2;i<argc;i+=2) {
        if(!allowed.contains(argv[i]) || !options.emplace(argv[i],argv[i+1]).second) throw std::runtime_error("Unknown or duplicate option.");
    }
    const auto config=loadMatchingConfig(options.contains(L"--config")?std::filesystem::path(options.at(L"--config")):std::filesystem::path{});
    auto settings=config.thresholds;
    double threshold=settings.appearance;
    if(command!=L"enroll"&&options.contains(L"--threshold")) {
        const auto& value=required(options,L"--threshold"); std::size_t consumed=0;
        threshold=std::stod(value,&consumed);
        if(consumed!=value.size() || !std::isfinite(threshold) || threshold < -1 || threshold > 1) throw std::runtime_error("Threshold must be finite and within [-1,1].");
    }
    // Validate required options before constructing the inference session.
    if(command==L"compare") {
        for(auto key:{L"--reference",L"--reference-roi",L"--query",L"--query-roi"}) required(options,key);
    } else {
        required(options,L"--image"); required(options,L"--roi");
        required(options,command==L"enroll"?L"--out":L"--template");
    }
    if(command==L"enroll"&&std::filesystem::exists(std::filesystem::path(required(options,L"--out"))))
        throw std::runtime_error("Template already exists; choose a new output path.");
    const auto model=options.contains(L"--model")?std::filesystem::path(options.at(L"--model")):std::filesystem::path(L"models/resnet18_embedding.onnx");
    const auto matcher=options.contains(L"--matcher-dir")?std::filesystem::path(options.at(L"--matcher-dir")):std::filesystem::path{};
    settings.appearance=threshold;
    RecommendedPipeline pipeline(model,matcher,settings);
    auto extract=[&](const Image& image,const Roi& roi,const wchar_t* option){
        if(options.contains(option)){auto mask=load(images,options.at(option));return pipeline.extract(image,roi,&mask);}
        return pipeline.extract(image,roi);
    };
    if(command==L"enroll") {
        auto image=load(images,required(options,L"--image"));
        auto record=extract(image,parseRoi(required(options,L"--roi"),image),L"--mask");
        saveEmbedding(std::filesystem::path(required(options,L"--out")),record);
        std::cout << "ENROLLED version=5 pipeline=" << recommendedPolicyId << " dimension=512 normalized=true local_features=" << record.local.features.keypoints.size()
            << " crop=" << record.roi.x << ',' << record.roi.y << ',' << record.roi.width << ',' << record.roi.height
            << " alignment=" << (record.local.features.keypoints.size()>=settings.minimumInliers?"READY":"INSUFFICIENT_FEATURES") << "\n";
        return 0;
    }
    EmbeddingTemplate reference,query;
    if(command==L"verify") {
        reference=loadEmbedding(std::filesystem::path(required(options,L"--template")));pipeline.validateTemplate(reference);
        auto image=load(images,required(options,L"--image"));query=extract(image,parseRoi(required(options,L"--roi"),image),L"--mask");
    } else {
        auto ref=load(images,required(options,L"--reference"));auto image=load(images,required(options,L"--query"));
        reference=extract(ref,parseRoi(required(options,L"--reference-roi"),ref),L"--reference-mask");
        query=extract(image,parseRoi(required(options,L"--query-roi"),image),L"--query-mask");
    }
    const double score=cosineSimilarity(reference.embedding,query.embedding);
    const auto alignment=pipeline.compare(reference,query);
    const bool appearance=score>=threshold;
    const bool accepted=appearance && alignment.accepted;
    std::cout << std::setprecision(10) << "appearance_similarity=" << score << " threshold=" << threshold
        << " appearance_match=" << (appearance?"true":"false")
        << " alignment=" << alignment.reason << " matches=" << alignment.matches << " inliers=" << alignment.inliers
        << " inlier_ratio=" << alignment.ratio << " minimum_inliers=" << settings.minimumInliers << " minimum_inlier_ratio=" << settings.minimumInlierRatio
        << " ransac_reprojection_ratio=" << settings.ransacReprojectionRatio
        << " lightglue_layers=" << alignment.matcherLayers << " pipeline=" << recommendedPolicyId
        << " decision=" << (accepted?"ACCEPT":"REJECT") << "\n";
    return accepted?0:4;
}
}
