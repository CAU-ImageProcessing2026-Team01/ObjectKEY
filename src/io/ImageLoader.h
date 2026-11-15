#pragma once
#include "core/Interfaces.h"

namespace objectkey {
class ImageLoader final : public IImageSource {
public:
    Result<Image> load(const std::filesystem::path& path) const override;
    static const char* backendName();
};
}
