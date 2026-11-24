#include "app/MatchingConfig.h"
#include <Windows.h>
#include <charconv>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
namespace objectkey {
namespace {
std::string trim(const std::string& value) {
    auto first=value.find_first_not_of(" \t\r\n");
    if(first==std::string::npos)return {};
    return value.substr(first,value.find_last_not_of(" \t\r\n")-first+1);
}
std::filesystem::path locate() {
    wchar_t buffer[32768]{};const auto size=GetModuleFileNameW(nullptr,buffer,32768);
    if(!size||size>=32768)throw std::runtime_error("Cannot locate the executable for matching settings.");
    auto directory=std::filesystem::path(buffer).parent_path();
    while(!directory.empty()) {
        auto candidate=directory/L"config/matching.ini";
        if(std::filesystem::exists(candidate))return candidate;
        auto parent=directory.parent_path();if(parent==directory)break;directory=parent;
    }
    throw std::runtime_error("Cannot find config/matching.ini beside the executable or in its ancestors. Restore the file or specify --config PATH.");
}
template<class T> T number(const std::map<std::string,std::string>& values,const char* key) {
    auto found=values.find(key);
    if(found==values.end())throw std::runtime_error(std::string("Missing matching setting: ")+key);
    T result{};const auto& value=found->second;
    auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
    if(parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size())
        throw std::runtime_error(std::string("Invalid numeric matching setting: ")+key);
    return result;
}
}
MatchingConfig loadMatchingConfig(const std::filesystem::path& explicitPath) {
    MatchingConfig config;config.path=std::filesystem::absolute(explicitPath.empty()?locate():explicitPath).lexically_normal();
    if(!std::filesystem::is_regular_file(config.path))throw std::runtime_error("Matching configuration is not a regular file.");
    if(std::filesystem::file_size(config.path)>16384)throw std::runtime_error("Matching configuration exceeds 16 KiB.");
    std::ifstream in(config.path,std::ios::binary);
    if(!in)throw std::runtime_error("Cannot open matching configuration.");
    std::map<std::string,std::string> values;std::string line;size_t lineNumber=0;
    while(std::getline(in,line)) {
        if(++lineNumber==1&&line.starts_with("\xEF\xBB\xBF"))line.erase(0,3);
        line=trim(line.substr(0,line.find_first_of("#;")));if(line.empty())continue;
        auto split=line.find('=');
        if(split==std::string::npos)throw std::runtime_error("Expected key=value at matching config line "+std::to_string(lineNumber));
        auto key=trim(line.substr(0,split)),value=trim(line.substr(split+1));
        if(key!="appearance_threshold"&&key!="minimum_inliers"&&key!="minimum_inlier_ratio"&&key!="ransac_reprojection_ratio")
            throw std::runtime_error("Unknown matching setting: "+key);
        if(!values.emplace(key,value).second)throw std::runtime_error("Duplicate matching setting: "+key);
    }
    if(!in.eof())throw std::runtime_error("Cannot read matching configuration.");
    config.thresholds.appearance=number<double>(values,"appearance_threshold");
    config.thresholds.minimumInliers=number<size_t>(values,"minimum_inliers");
    config.thresholds.minimumInlierRatio=number<double>(values,"minimum_inlier_ratio");
    config.thresholds.ransacReprojectionRatio=number<double>(values,"ransac_reprojection_ratio");
    config.thresholds.validate();return config;
}
}
