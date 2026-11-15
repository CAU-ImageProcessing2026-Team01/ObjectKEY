#include "app/Application.h"
#include "io/ImageLoader.h"
#include <fstream>
#include <iostream>
#include <string_view>

namespace objectkey {
namespace {
void help() {
    std::cout << "ObjectKEY - image authentication scaffold\n"
        "Usage:\n  ObjectKEY.exe --help\n  ObjectKEY.exe inspect <image>\n"
        "  ObjectKEY.exe enroll <image>\n  ObjectKEY.exe verify <image>\n"
        "  ObjectKEY.exe evaluate <manifest.txt>\n\n"
        "enroll/verify/evaluate validate inputs only; authentication is NOT implemented.\n"
        "Manifest: UTF-8, one image path per line, relative to manifest directory.\n"
        "Exit codes: 0=input inspection/help success, 2=invalid input, 3=not implemented.\n";
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
int Application::enroll(const std::filesystem::path& path) const {
    if (!inspect(path)) return 2;
    std::cout << "NOT_IMPLEMENTED: enrollment needs ROI, preprocessing, feature extraction and template storage. No template saved.\n";
    return 3;
}
int Application::verify(const std::filesystem::path& path) const {
    if (!inspect(path)) return 2;
    const AuthResult result{};
    std::cout << "NOT_IMPLEMENTED decision=NOT_EVALUATED: " << result.reason << " No access granted.\n";
    return 3;
}
int Application::evaluate(const std::filesystem::path& manifest) const {
    std::ifstream input(manifest, std::ios::binary);
    if (!input) { std::cerr << "INPUT_ERROR: cannot open manifest.\n"; return 2; }
    std::string line;
    std::size_t count = 0, failures = 0, lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        ++count;
        std::cout << "Sample line " << lineNumber << ": ";
        try {
            const auto utf8 = std::u8string(reinterpret_cast<const char8_t*>(line.data()), line.size());
            auto path = std::filesystem::path(utf8);
            if (path.is_relative()) path = manifest.parent_path() / path;
            if (!inspect(path)) ++failures;
        } catch (const std::exception&) { ++failures; std::cerr << "INPUT_ERROR: invalid path.\n"; }
    }
    if (input.bad() || count == 0) { std::cerr << "INPUT_ERROR: unreadable or empty manifest.\n"; return 2; }
    std::cout << "INPUT_SUMMARY samples=" << count << " failures=" << failures << '\n';
    if (failures != 0) return 2;
    std::cout << "NOT_IMPLEMENTED: evaluation requires templates, labeled samples and authentication. No accuracy computed.\n";
    return 3;
}
int Application::run(int argc, wchar_t* argv[]) const {
    if (argc == 1 || (argc == 2 && (std::wstring_view(argv[1]) == L"--help" || std::wstring_view(argv[1]) == L"-h"))) { help(); return 0; }
    if (argc != 3) { help(); return 2; }
    const std::wstring_view command(argv[1]);
    std::cout << "Image backend: " << ImageLoader::backendName() << '\n';
    const std::filesystem::path path(argv[2]);
    if (command == L"inspect") return inspect(path) ? 0 : 2;
    if (command == L"enroll") return enroll(path);
    if (command == L"verify") return verify(path);
    if (command == L"evaluate") return evaluate(path);
    std::cerr << "INPUT_ERROR: unknown command.\n"; help(); return 2;
}
}
