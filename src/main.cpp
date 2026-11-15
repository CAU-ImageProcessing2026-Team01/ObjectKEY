#include "app/Application.h"
#include "io/ImageLoader.h"
#include <exception>
#include <iostream>

int wmain(int argc, wchar_t* argv[]) {
    try {
        const objectkey::ImageLoader images;
        return objectkey::Application(images).run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
