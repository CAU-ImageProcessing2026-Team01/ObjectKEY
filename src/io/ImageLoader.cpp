#include "io/ImageLoader.h"
#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>


namespace objectkey {
namespace {
Result<Image> failure(const std::string& message) { return {{Status::IoError, message}, std::nullopt}; }
constexpr std::uint64_t maxPixels = 40'000'000;
}
const char* ImageLoader::backendName() {
    return "Windows Imaging Component (WIC)";

}
Result<Image> ImageLoader::load(const std::filesystem::path& path) const {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return failure("Image file does not exist or is not a regular file.");
    const auto fileSize = std::filesystem::file_size(path, ec);
    if (ec || fileSize == 0 || fileSize > 256ULL * 1024 * 1024) return failure("Image file is empty, inaccessible, or exceeds 256 MiB.");
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(init) && init != RPC_E_CHANGED_MODE) return failure("Cannot initialize COM for image decoding.");
    struct ComScope { bool owns; ~ComScope() { if (owns) CoUninitialize(); } } scope{SUCCEEDED(init)};
    using Microsoft::WRL::ComPtr;
    ComPtr<IWICImagingFactory> factory;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) return failure("Cannot create WIC decoder.");
    ComPtr<IWICBitmapDecoder> decoder;
    if (FAILED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder))) return failure("Unsupported, inaccessible, or corrupt image.");
    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(decoder->GetFrame(0, &frame))) return failure("Cannot read first image frame.");
    UINT width = 0, height = 0;
    if (FAILED(frame->GetSize(&width, &height)) || width == 0 || height == 0 || static_cast<std::uint64_t>(width) * height > maxPixels) return failure("Invalid image size or image exceeds 40 million pixels.");
    ComPtr<IWICFormatConverter> converter;
    if (FAILED(factory->CreateFormatConverter(&converter)) || FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) return failure("Cannot convert image to BGR8.");
    Image result{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 3)};
    if (FAILED(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(result.pixels.size()), result.pixels.data()))) return failure("Cannot decode image pixels.");
    return {{Status::Ok, "Image loaded."}, std::move(result)};

}
}
