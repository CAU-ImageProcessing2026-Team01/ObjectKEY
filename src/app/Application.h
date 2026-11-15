#pragma once
#include "core/Interfaces.h"

namespace objectkey {
class Application {
public:
    explicit Application(const IImageSource& images) : images_(images) {}
    int run(int argc, wchar_t* argv[]) const;
private:
    const IImageSource& images_;
    int enroll(const std::filesystem::path&) const;
    int verify(const std::filesystem::path&) const;
    int evaluate(const std::filesystem::path&) const;
    bool inspect(const std::filesystem::path&) const;
};
}
