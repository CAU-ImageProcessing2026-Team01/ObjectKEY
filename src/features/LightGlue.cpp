#include "features/LightGlue.h"
#include "features/ResNet18.h"
#include "vision/Sift.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace objectkey {
namespace {
struct Features {
    std::vector<float> descriptors,encoding;
    std::vector<size_t> indices;
};
Ort::Value tensor(std::vector<float>& data,const std::vector<int64_t>& shape) {
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory,data.data(),data.size(),shape.data(),shape.size());
}
std::vector<float> floats(const Ort::Value& value) {
    if(!value.IsTensor()||value.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Invalid LightGlue model output.");
    auto data=value.GetTensorData<float>();size_t n=value.GetTensorTypeAndShapeInfo().GetElementCount();
    for(size_t i=0;i<n;i++)if(!std::isfinite(data[i]))throw std::runtime_error("Non-finite LightGlue output.");
    return {data,data+n};
}
void prune(Features& f,const float* confidence,const float* matchability,float threshold) {
    std::vector<size_t> keep;const size_t n=f.indices.size();
    for(size_t i=0;i<n;i++)if(matchability[i]>.01f||confidence[i]<=threshold)keep.push_back(i);
    Features result;result.indices.reserve(keep.size());result.descriptors.reserve(keep.size()*256);result.encoding.reserve(keep.size()*128);
    for(auto i:keep){result.indices.push_back(f.indices[i]);result.descriptors.insert(result.descriptors.end(),f.descriptors.begin()+i*256,f.descriptors.begin()+(i+1)*256);}
    for(size_t plane=0;plane<2;plane++)for(auto i:keep)
        result.encoding.insert(result.encoding.end(),f.encoding.begin()+(plane*n+i)*64,f.encoding.begin()+(plane*n+i+1)*64);
    f=std::move(result);
}
}
struct LightGlue::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"ObjectKEY-LightGlue"};
    Ort::SessionOptions options;
    Ort::Session initialize{nullptr};
    std::vector<Ort::Session> layers;
    std::string hash;
    explicit Impl(const std::filesystem::path& directory) {
        std::ifstream manifest(directory/"manifest.txt");std::string version;int count=0;double depth=0,width=0,filter=0;
        if(!(manifest>>version>>hash>>count>>depth>>width>>filter)||version!="OBJECTKEY_LIGHTGLUE_SIFT_V1"||
           hash!="5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0"||count!=9||depth!=.95||width!=.99||filter!=.1)
            throw std::runtime_error("Missing or incompatible LightGlue models. Run setup.ps1.");
        options.SetIntraOpNumThreads(1);options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        for(int i=-1;i<9;i++) {
            std::string filename,digest;const auto expected=i<0?"initialize.onnx":"layer"+std::to_string(i)+".onnx";
            if(!(manifest>>filename>>digest)||filename!=expected||fileSha256(directory/filename)!=digest)
                throw std::runtime_error("LightGlue model fingerprint mismatch. Re-run config/setup/export-lightglue.py.");
            Ort::Session session(env,(directory/filename).c_str(),options);
            if(session.GetInputCount()!=(i<0?2:4)||session.GetOutputCount()!=(i<0?2:7))throw std::runtime_error("Unexpected LightGlue model interface.");
            if(i<0)initialize=std::move(session);else layers.push_back(std::move(session));
        }
        std::string extra;if(manifest>>extra)throw std::runtime_error("Unexpected LightGlue manifest data.");
    }
    Features prepare(const LocalFeatures& local) {
        const size_t n=local.features.keypoints.size();std::vector<float> descriptors=local.features.descriptors,positions(n*4);
        vision::rootSift(descriptors);
        for(size_t i=0;i<n;i++) {
            const auto& p=local.features.keypoints[i];float scale=float(std::max(local.width,local.height));
            positions[i*4]=(2*p.x-.5f-local.width)/scale;positions[i*4+1]=(2*p.y-.5f-local.height)/scale;
            positions[i*4+2]=2*p.scale;positions[i*4+3]=p.orientation;
        }
        std::vector<Ort::Value> input;input.push_back(tensor(descriptors,{1,int64_t(n),128}));input.push_back(tensor(positions,{1,int64_t(n),4}));
        const char* names[]={"descriptors","positions"};const char* outputs[]={"projected","encoding"};
        auto values=initialize.Run(Ort::RunOptions{nullptr},names,input.data(),2,outputs,2);
        Features result{floats(values[0]),floats(values[1]),std::vector<size_t>(n)};
        if(result.descriptors.size()!=n*256||result.encoding.size()!=n*128)throw std::runtime_error("Unexpected LightGlue feature dimensions.");
        std::iota(result.indices.begin(),result.indices.end(),0);return result;
    }
};
LightGlue::LightGlue(const std::filesystem::path& directory):impl_(std::make_unique<Impl>(directory)){}
LightGlue::~LightGlue()=default;
const std::string& LightGlue::fingerprint() const{return impl_->hash;}
FeatureMatches LightGlue::match(const LocalFeatures& reference,const LocalFeatures& query,unsigned* usedLayers) {
    validateSpatialFeatures(reference);validateSpatialFeatures(query);
    if(usedLayers)*usedLayers=0;
    if(reference.features.keypoints.empty()||query.features.keypoints.empty())return {};
    auto a=impl_->prepare(reference),b=impl_->prepare(query);const size_t initialCount=a.indices.size()+b.indices.size();
    for(size_t layer=0;layer<9;layer++) {
        const size_t n=a.indices.size(),m=b.indices.size();if(!n||!m)return {};
        std::vector<Ort::Value> inputs;
        inputs.push_back(tensor(a.descriptors,{1,int64_t(n),256}));inputs.push_back(tensor(b.descriptors,{1,int64_t(m),256}));
        inputs.push_back(tensor(a.encoding,{2,1,1,int64_t(n),64}));inputs.push_back(tensor(b.encoding,{2,1,1,int64_t(m),64}));
        const char* names[]={"desc0","desc1","encoding0","encoding1"};
        const char* outputNames[]={"next0","next1","confidence0","confidence1","matchability0","matchability1","scores"};
        auto output=impl_->layers[layer].Run(Ort::RunOptions{nullptr},names,inputs.data(),4,outputNames,7);
        const std::array<size_t,7> sizes{n*256,m*256,n,m,n,m,n*m};
        for(size_t i=0;i<7;i++)if(output[i].GetTensorTypeAndShapeInfo().GetElementCount()!=sizes[i])throw std::runtime_error("Unexpected adaptive LightGlue output shape.");
        const auto c0=output[2].GetTensorData<float>(),c1=output[3].GetTensorData<float>();
        const float threshold=float(.8+.1*std::exp(-4.*layer/9.));size_t uncertain=0;
        for(size_t i=0;i<n;i++)uncertain+=c0[i]<threshold;for(size_t i=0;i<m;i++)uncertain+=c1[i]<threshold;
        // Official stopping denominator includes pruned points, which count as confident.
        const float ratio=1.f-float(uncertain)/float(initialCount);
        if(usedLayers)*usedLayers=unsigned(layer+1);
        if(layer==8||ratio>.95f) {
            auto scores=floats(output[6]);std::vector<size_t> best0(n),best1(m);
            for(size_t i=0;i<n;i++)best0[i]=size_t(std::max_element(scores.begin()+i*m,scores.begin()+(i+1)*m)-(scores.begin()+i*m));
            for(size_t j=0;j<m;j++){size_t best=0;for(size_t i=1;i<n;i++)if(scores[i*m+j]>scores[best*m+j])best=i;best1[j]=best;}
            FeatureMatches result;
            for(size_t i=0;i<n;i++){auto j=best0[i];if(best1[j]==i&&std::exp(scores[i*m+j])>.1f)result.emplace_back(a.indices[i],b.indices[j]);}
            std::sort(result.begin(),result.end());return result;
        }
        a.descriptors=floats(output[0]);b.descriptors=floats(output[1]);
        prune(a,c0,output[4].GetTensorData<float>(),threshold);prune(b,c1,output[5].GetTensorData<float>(),threshold);
    }
    throw std::runtime_error("Invalid LightGlue inference state.");
}
}
