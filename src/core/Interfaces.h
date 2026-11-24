#pragma once
#include "core/Types.h"

namespace objectkey {
class IImageSource {
public:
    virtual ~IImageSource() = default;
    virtual Result<Image> load(const std::filesystem::path& path) const = 0;
};
}
