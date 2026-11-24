#pragma once
#include "core/Interfaces.h"

namespace objectkey {
class Application {
public:
    explicit Application(const IImageSource& images) : images_(images) {}
    int run(int argc, wchar_t* argv[]) const;
private:
    const IImageSource& images_;


    bool inspect(const std::filesystem::path&) const;
};
}
