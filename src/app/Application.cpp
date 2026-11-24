#include "app/Application.h"
#include "app/EmbeddingCommands.h"
#include "io/ImageLoader.h"
#include <iostream>
#include <string_view>

namespace objectkey {
namespace {
void help() {
    std::cout << "ObjectKEY - GrabCut, ResNet-18 and SIFT/LightGlue inlier matching\n"
        "Usage:\n"
        "  ObjectKEY.exe --gui\n"
        "  ObjectKEY.exe inspect <image>\n"
        "  ObjectKEY.exe enroll --image PATH --roi x,y,w,h --out TEMPLATE\n"
        "  ObjectKEY.exe verify --image PATH --roi x,y,w,h --template TEMPLATE [--threshold VALUE]\n"
        "  ObjectKEY.exe compare --reference PATH --reference-roi x,y,w,h --query PATH --query-roi x,y,w,h [--threshold VALUE]\n"
        "ROI can be 'full'. GUI: load images, drag ROIs, Extract, then Compare. Save is optional.\n"
        "Optional masks: --mask PATH for enroll/verify; --reference-mask and --query-mask for compare.\n"
        "Without a mask, ROI-initialized GrabCut runs automatically. Masks must match image dimensions.\n"
        "Optional --model ONNX and --matcher-dir DIR; defaults: models/resnet18_embedding.onnx and models/lightglue_sift.\n"
        "Settings: config/matching.ini beside the executable or in an ancestor; override with --config PATH.\n"
        "--threshold overrides appearance for this command. GUI Reload Settings refreshes read-only values.\n"
        "Default settings: appearance >=0.8, >=10 affine inliers, ratio >=0.70, RANSAC distance ratio=0.025.\n"
        "These image-space checks do not estimate physical camera angles. V1-V4 templates require re-enrollment.\n"
        "Exit codes: 0=success/accept, 2=error, 4=reject.\n";
}
}
bool Application::inspect(const std::filesystem::path& path) const {
    const auto result = images_.load(path);
    if (!result.ok()) { std::cerr << "INPUT_ERROR: " << result.outcome.message << '\n'; return false; }
    const auto& image = *result.value;
    std::cout << "IMAGE_LOADED width=" << image.width << " height=" << image.height
        << " channels=3 format=BGR8 bytes=" << image.pixels.size() << '\n';
    return true;
}
int Application::run(int argc, wchar_t* argv[]) const {
    if (argc == 1 || (argc == 2 && (std::wstring_view(argv[1]) == L"--help" || std::wstring_view(argv[1]) == L"-h"))) { help(); return 0; }
    if (argc >= 2 && (std::wstring_view(argv[1]) == L"enroll" || std::wstring_view(argv[1]) == L"verify" || std::wstring_view(argv[1]) == L"compare")) return runEmbeddingCommand(argc, argv, images_);
    if (argc != 3) { help(); return 2; }
    const std::wstring_view command(argv[1]);
    std::cout << "Image backend: " << ImageLoader::backendName() << '\n';
    const std::filesystem::path path(argv[2]);
    if (command == L"inspect") return inspect(path) ? 0 : 2;


    std::cerr << "INPUT_ERROR: unknown command.\n"; help(); return 2;
}
}
