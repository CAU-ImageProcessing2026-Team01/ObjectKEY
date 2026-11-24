#include "features/ResNet18.h"
#include <onnxruntime_cxx_api.h>
#include <Windows.h>
#include <bcrypt.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace objectkey {
std::string fileSha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open model file. Run setup.ps1 first.");
    struct Handles {
        BCRYPT_ALG_HANDLE algorithm=nullptr;
        BCRYPT_HASH_HANDLE hash=nullptr;
        ~Handles() { if(hash) BCryptDestroyHash(hash); if(algorithm) BCryptCloseAlgorithmProvider(algorithm,0); }
    } handles;
    if (BCryptOpenAlgorithmProvider(&handles.algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0 ||
        BCryptCreateHash(handles.algorithm,&handles.hash,nullptr,0,nullptr,0,0)<0) throw std::runtime_error("Cannot initialize SHA256.");
    std::array<char,65536> block;
    while(input) {
        input.read(block.data(),block.size());
        if(input.gcount() && BCryptHashData(handles.hash,reinterpret_cast<PUCHAR>(block.data()),static_cast<ULONG>(input.gcount()),0)<0)
            throw std::runtime_error("Cannot hash ONNX model.");
    }
    if(input.bad()) throw std::runtime_error("Cannot read ONNX model.");
    std::array<unsigned char,32> digest{};
    if(BCryptFinishHash(handles.hash,digest.data(),static_cast<ULONG>(digest.size()),0)<0) throw std::runtime_error("Cannot finish SHA256.");
    std::ostringstream out;
    for(auto value:digest) out << std::hex << std::setw(2) << std::setfill('0') << int(value);
    return out.str();
}
struct ResNet18::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"ObjectKEY"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    std::string hash;
    explicit Impl(const std::filesystem::path& model) : hash(fileSha256(model)) {
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session=Ort::Session(env,model.c_str(),options);
        if(session.GetInputCount()!=1 || session.GetOutputCount()!=1) throw std::runtime_error("Expected one model input and output.");
        const auto inputInfo=session.GetInputTypeInfo(0);
        const auto outputInfo=session.GetOutputTypeInfo(0);
        if(inputInfo.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           outputInfo.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           inputInfo.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,3,224,224} ||
           outputInfo.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,512})
            throw std::runtime_error("Expected float model [1,3,224,224] -> [1,512].");
    }
};
ResNet18::ResNet18(const std::filesystem::path& model):impl_(std::make_unique<Impl>(model)) {}
ResNet18::~ResNet18()=default;
const std::string& ResNet18::modelHash() const { return impl_->hash; }
Embedding ResNet18::extract(const Image& image,const Roi& roi) {
    auto pixels=preprocessResNet(image,roi);
    const std::array<int64_t,4> shape{1,3,224,224};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(memory,pixels.data(),pixels.size(),shape.data(),shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName=impl_->session.GetInputNameAllocated(0,allocator);
    auto outputName=impl_->session.GetOutputNameAllocated(0,allocator);
    const char* inputs[]={inputName.get()}; const char* outputs[]={outputName.get()};
    auto values=impl_->session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
    if(!values[0].IsTensor() || values[0].GetTensorTypeAndShapeInfo().GetElementCount()!=512) throw std::runtime_error("Unexpected inference output.");
    Embedding embedding;
    std::copy_n(values[0].GetTensorData<float>(),embedding.size(),embedding.begin());
    normalizeEmbedding(embedding);
    return embedding;
}
}
