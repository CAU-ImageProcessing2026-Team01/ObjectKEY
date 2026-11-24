#include "app/Application.h"
#include "io/ImageLoader.h"
#include "gui/DesktopApp.h"
#include <Windows.h>
#include <exception>
#include <iostream>
#include <string_view>

int wmain(int argc, wchar_t* argv[]) {
    const bool gui=argc==1 || (argc==2 && std::wstring_view(argv[1])==L"--gui");
    try {
        if(gui) {
            DWORD processes[2]{};
            if(GetConsoleProcessList(processes,2)==1) FreeConsole();
            return objectkey::runGui();
        }
        const objectkey::ImageLoader images;
        return objectkey::Application(images).run(argc, argv);
    } catch(const std::exception& error) {
        if(gui) MessageBoxA(nullptr,error.what(),"ObjectKEY startup error",MB_OK|MB_ICONERROR);
        else std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
