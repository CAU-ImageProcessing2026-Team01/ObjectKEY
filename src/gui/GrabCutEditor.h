#pragma once
#include "core/Types.h"
#include <Windows.h>
#include <optional>
namespace objectkey {
// Modal foreground editor. Cancel returns no mask and never changes the registration.
std::optional<Image> editGrabCutMask(HWND owner,const Image& image,const Roi& roi);
}
