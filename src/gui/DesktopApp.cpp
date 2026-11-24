#include "gui/DesktopApp.h"
#include "gui/GrabCutEditor.h"
#include "gui/RoiSelection.h"
#include "app/MatchingConfig.h"
#include "features/RecommendedPipeline.h"
#include "io/ImageLoader.h"
#include <Windows.h>
#include <windowsx.h>
#include <shobjidl.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace objectkey {
namespace {
constexpr int LoadReference=101,LoadTemplate=102,LoadQuery=103,Compare=104,Extract=105,SaveTemplate=106;
constexpr int ReloadSettings=111,ReferenceFull=112,QueryFull=113;
constexpr int ReferenceEdit=114,QueryEdit=115,ReferenceMask=116,QueryMask=117;
constexpr gui::DisplayRect panes[2]={{24,216,440,330},{490,216,440,330}};
struct ImageInput {
    Image image;
    std::filesystem::path path;
    Roi roi;
    std::optional<Image> suppliedMask;
};
struct TaskResult {
    bool registration=false;
    MatchingThresholds settings;
    EmbeddingTemplate record;
    double score=0;
    AlignmentResult alignment;
};
struct State {
    HWND window{},status{},result{},settingsLabel{},labels[2]{},roiLabels[2]{};
    HFONT font{};
    MatchingConfig config;
    std::filesystem::path model;
    ImageInput inputs[2];
    std::optional<EmbeddingTemplate> reference;
    Image templateMask;
    std::filesystem::path templatePath;
    std::future<TaskResult> task;
    bool busy=false;
    int dragging=-1;
    POINT anchor{},cursor{};
};
std::wstring widen(const std::string& value) {
    if(value.empty())return {};
    int count=MultiByteToWideChar(CP_UTF8,0,value.data(),int(value.size()),nullptr,0);
    std::wstring text(count,L' ');MultiByteToWideChar(CP_UTF8,0,value.data(),int(value.size()),text.data(),count);return text;
}
std::filesystem::path locateModel() {
    wchar_t buffer[32768]{};GetModuleFileNameW(nullptr,buffer,32768);
    auto directory=std::filesystem::path(buffer).parent_path();
    for(int i=0;i<6&&!directory.empty();i++) {
        auto path=directory/L"models/resnet18_embedding.onnx";
        if(std::filesystem::is_regular_file(path))return path;
        directory=directory.parent_path();
    }
    throw std::runtime_error("Cannot find models/resnet18_embedding.onnx. Run setup.ps1.");
}
std::optional<std::filesystem::path> choose(HWND owner,bool save,const wchar_t* title,const wchar_t* pattern) {
    using Microsoft::WRL::ComPtr;
    ComPtr<IFileDialog> dialog;
    HRESULT hr=save?CoCreateInstance(CLSID_FileSaveDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog)):
        CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog));
    if(FAILED(hr))throw std::runtime_error("Cannot open Windows file dialog.");
    COMDLG_FILTERSPEC filters[]={{L"Supported files",pattern},{L"All files",L"*.*"}};
    dialog->SetFileTypes(2,filters);dialog->SetTitle(title);
    FILEOPENDIALOGOPTIONS flags{};dialog->GetOptions(&flags);
    dialog->SetOptions(flags|FOS_FORCEFILESYSTEM|FOS_PATHMUSTEXIST|(save?FOS_OVERWRITEPROMPT:FOS_FILEMUSTEXIST));
    if(save){dialog->SetDefaultExtension(L"okt");dialog->SetFileName(L"object.okt");}
    hr=dialog->Show(owner);
    if(hr==HRESULT_FROM_WIN32(ERROR_CANCELLED))return std::nullopt;
    if(FAILED(hr))throw std::runtime_error("File selection failed.");
    ComPtr<IShellItem> item;if(FAILED(dialog->GetResult(&item)))throw std::runtime_error("Cannot read file selection.");
    PWSTR path=nullptr;if(FAILED(item->GetDisplayName(SIGDN_FILESYSPATH,&path)))throw std::runtime_error("Cannot read selected path.");
    std::filesystem::path result(path);CoTaskMemFree(path);return result;
}
Image loadImage(const std::filesystem::path& path) {
    auto result=ImageLoader{}.load(path);if(!result.ok())throw std::runtime_error(result.outcome.message);return std::move(*result.value);
}
Image maskImage(const LocalFeatures& local) {
    Image result{local.width,local.height,{}};result.pixels.reserve(local.objectMask.size()*3);
    for(auto value:local.objectMask)for(int c=0;c<3;c++)result.pixels.push_back(value?255:0);return result;
}
void updateControls(State& s) {
    auto enable=[&](int id,bool condition){EnableWindow(GetDlgItem(s.window,id),!s.busy&&condition);};
    for(int id:{LoadReference,LoadTemplate,LoadQuery,ReloadSettings})enable(id,true);
    enable(Extract,!s.inputs[0].image.pixels.empty());
    enable(SaveTemplate,s.reference.has_value());
    enable(Compare,s.reference.has_value()&&!s.inputs[1].image.pixels.empty());
    for(int i=0;i<2;i++)for(int id:{i?QueryFull:ReferenceFull,i?QueryEdit:ReferenceEdit,i?QueryMask:ReferenceMask})
        enable(id,!s.inputs[i].image.pixels.empty());
}
void showSettings(State& s) {
    const auto& t=s.config.thresholds;std::wostringstream text;
    text << L"Appearance >= " << std::setprecision(10) << t.appearance << L"   |   Inliers >= " << t.minimumInliers
        << L"   |   Ratio >= " << t.minimumInlierRatio << L"   |   RANSAC: " << t.ransacReprojectionRatio;
    SetWindowTextW(s.settingsLabel,text.str().c_str());
}
void showRoi(State& s,int side) {
    const auto& input=s.inputs[side];std::wostringstream text;
    if(input.image.pixels.empty())text << (side==0&&s.reference?L"Saved template: load its source image to change ROI.":L"Load an image to select its ROI.");
    else text << L"ROI: " << input.roi.x << L", " << input.roi.y << L", " << input.roi.width << L", " << input.roi.height
        << (input.suppliedMask?L"   |   Custom mask":L"   |   Automatic GrabCut");
    SetWindowTextW(s.roiLabels[side],text.str().c_str());
}
void changed(State& s,int side,const wchar_t* message) {
    if(side==0){s.reference.reset();s.templateMask={};s.templatePath.clear();}
    SetWindowTextW(s.result,message);showRoi(s,side);updateControls(s);InvalidateRect(s.window,nullptr,FALSE);
}
void setRoi(State& s,int side,const Roi& roi) {
    auto& input=s.inputs[side];input.roi=roi;input.suppliedMask.reset();
    changed(s,side,side==0?L"Reference ROI changed. Click Extract before comparing. Saving is optional.":L"Query ROI changed. Click Compare to update the result.");
    SetWindowTextW(s.status,L"ROI selected. Automatic GrabCut will run when you extract or compare.");
}
void loadPhoto(State& s,int side) {
    auto path=choose(s.window,false,side?L"Load Query Image":L"Load Reference Image",L"*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff");
    if(!path)return;
    auto image=loadImage(*path);ImageInput input;input.roi={0,0,image.width,image.height};input.image=std::move(image);input.path=*path;
    s.inputs[side]=std::move(input);
    SetWindowTextW(s.labels[side],((side?L"Query: ":L"Reference: ")+path->filename().wstring()).c_str());
    changed(s,side,side?L"Query image loaded. Drag around the object, then click Compare.":L"Reference image loaded. Drag around the object, then click Extract. Saving is optional.");
    SetWindowTextW(s.status,L"Image loading does not extract features. Drag on either photo to select an ROI; Escape cancels a drag.");
}
void setMask(State& s,int side,bool editor) {
    auto& input=s.inputs[side];if(input.image.pixels.empty())return;
    std::optional<Image> mask;
    if(editor)mask=editGrabCutMask(s.window,input.image,input.roi);
    else {
        auto path=choose(s.window,false,L"Load Object Mask",L"*.png;*.bmp;*.tif;*.tiff");if(!path)return;mask=loadImage(*path);
        if(mask->width!=input.image.width||mask->height!=input.image.height)throw std::runtime_error("Mask dimensions must match the loaded image.");
        for(size_t k=0;k<mask->pixels.size();k+=3)if(mask->pixels[k]!=mask->pixels[k+1]||mask->pixels[k]!=mask->pixels[k+2])
            throw std::runtime_error("Mask must be grayscale: white object, black background.");
    }
    if(!mask)return;
    input.suppliedMask=std::move(mask);
    changed(s,side,side?L"Query mask changed. Click Compare.":L"Reference mask changed. Click Extract before comparing. Saving is optional.");
    SetWindowTextW(s.status,L"Custom mask ready. Drawing a new ROI or choosing Full Image restores automatic GrabCut.");
}
void begin(State& s,bool registration) {
    const int side=registration?0:1;
    if(s.inputs[side].image.pixels.empty())throw std::runtime_error("Load an image first.");
    if(!registration&&!s.reference)throw std::runtime_error("Extract a reference template or load an existing template first.");
    if(!registration)RecommendedPipeline::requireCurrentTemplate(*s.reference);
    const auto input=s.inputs[side];const auto reference=s.reference;const auto model=s.model;const auto settings=s.config.thresholds;
    s.task=std::async(std::launch::async,[input,reference,model,settings,registration] {
        TaskResult result;result.registration=registration;result.settings=settings;
        RecommendedPipeline pipeline(model,{},settings);
        if(!registration)pipeline.validateTemplate(*reference);
        auto record=pipeline.extract(input.image,input.roi,input.suppliedMask?&*input.suppliedMask:nullptr);
        if(registration)result.record=std::move(record);
        else {result.score=cosineSimilarity(reference->embedding,record.embedding);result.alignment=pipeline.compare(*reference,record);}
        return result;
    });
    s.busy=true;updateControls(s);SetWindowTextW(s.result,L"Processing...");
    SetWindowTextW(s.status,registration?L"Extracting the reference template into memory...":L"Extracting query features and comparing...");
}
void finish(State& s) {
    if(!s.busy||!s.task.valid()||s.task.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
    s.busy=false; // Clear before consuming the future: message boxes pump WM_TIMER.
    try {
        auto result=s.task.get();
        if(result.registration) {
            s.reference=std::move(result.record);s.templatePath.clear();s.templateMask=maskImage(s.reference->local);
            SetWindowTextW(s.result,L"Reference ready. Load a query image or click Compare. Saving is optional.");
            std::wostringstream detail;detail << L"Reference features: " << s.reference->local.features.keypoints.size()
                << L" | In memory only. Click Save to keep this template after closing the app.";
            if(s.reference->local.features.keypoints.size()<result.settings.minimumInliers)detail << L"\r\nToo few local features for the configured inlier requirement.";
            SetWindowTextW(s.status,detail.str().c_str());
        } else {
            bool appearance=result.score>=result.settings.appearance;std::wostringstream message,detail;
            message << (appearance&&result.alignment.accepted?L"ACCEPT":L"REJECT") << L" | " << (appearance?L"Appearance match":L"Appearance mismatch")
                << L" | Similarity " << std::fixed << std::setprecision(6) << result.score << L" | Threshold " << result.settings.appearance;
            detail << widen(result.alignment.reason) << L" | Inliers " << result.alignment.inliers << L"/" << result.alignment.matches
                << L" (min " << result.settings.minimumInliers << L") | Inlier ratio " << std::fixed << std::setprecision(4) << result.alignment.ratio
                << L" (min " << result.settings.minimumInlierRatio << L")\r\nDecision: appearance + inlier match.";
            SetWindowTextW(s.result,message.str().c_str());SetWindowTextW(s.status,detail.str().c_str());
        }
    }catch(const std::exception& error){SetWindowTextW(s.result,L"Processing failed. Adjust the image, ROI or mask and try again.");SetWindowTextW(s.status,widen(error.what()).c_str());MessageBoxW(s.window,widen(error.what()).c_str(),L"ObjectKEY Error",MB_OK|MB_ICONERROR);}
    updateControls(s);InvalidateRect(s.window,nullptr,FALSE);
}
RECT nativeRect(gui::DisplayRect box){return {box.x,box.y,box.x+box.width,box.y+box.height};}
void drawImage(HDC dc,const Image& image,gui::DisplayRect box,const wchar_t* empty) {
    auto bounds=nativeRect(box);HBRUSH background=CreateSolidBrush(RGB(240,243,247));FillRect(dc,&bounds,background);DeleteObject(background);
    if(image.pixels.empty()){SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(90,100,110));DrawTextW(dc,empty,-1,&bounds,DT_CENTER|DT_VCENTER|DT_SINGLELINE);return;}
    auto display=gui::fitImage(image.width,image.height,box);
    size_t stride=(size_t(image.width)*3+3)&~size_t(3);std::vector<unsigned char> padded(stride*image.height);
    for(unsigned row=0;row<image.height;row++)std::copy_n(image.pixels.data()+size_t(row)*image.width*3,size_t(image.width)*3,padded.data()+size_t(row)*stride);
    BITMAPINFO info{};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=LONG(image.width);info.bmiHeader.biHeight=-LONG(image.height);info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=24;info.bmiHeader.biCompression=BI_RGB;
    SetStretchBltMode(dc,HALFTONE);SetBrushOrgEx(dc,0,0,nullptr);
    StretchDIBits(dc,display.x,display.y,display.width,display.height,0,0,image.width,image.height,padded.data(),&info,DIB_RGB_COLORS,SRCCOPY);
}
void drawRoi(HDC dc,const ImageInput& input,gui::DisplayRect pane,const Roi& roi) {
    const int saved=SaveDC(dc);
    IntersectClipRect(dc,pane.x,pane.y,pane.x+pane.width,pane.y+pane.height);
    auto display=gui::fitImage(input.image.width,input.image.height,pane);
    const double sx=double(display.width)/input.image.width,sy=double(display.height)/input.image.height;
    HPEN pen=CreatePen(PS_SOLID,2,RGB(30,220,80));auto oldPen=SelectObject(dc,pen);auto oldBrush=SelectObject(dc,GetStockObject(HOLLOW_BRUSH));
    Rectangle(dc,display.x+int(roi.x*sx),display.y+int(roi.y*sy),display.x+int((roi.x+roi.width)*sx),display.y+int((roi.y+roi.height)*sy));
    SelectObject(dc,oldPen);SelectObject(dc,oldBrush);DeleteObject(pen);RestoreDC(dc,saved);
}
void paint(State& s,HDC dc) {
    for(int side=0;side<2;side++) {
        const auto& input=s.inputs[side];
        const auto& image=input.image.pixels.empty()&&side==0?s.templateMask:input.image;
        drawImage(dc,image,panes[side],L"Load an image to begin");
        if(!input.image.pixels.empty()) {
            auto roi=input.roi;
            if(s.dragging==side) {
                auto selected=gui::dragRoi(input.image.width,input.image.height,gui::fitImage(input.image.width,input.image.height,panes[side]),s.anchor.x,s.anchor.y,s.cursor.x,s.cursor.y);
                if(selected)roi=*selected;
            }
            drawRoi(dc,input,panes[side],roi);
        }
    }
}
void cancelDrag(State& s){if(s.dragging>=0){s.dragging=-1;ReleaseCapture();InvalidateRect(s.window,nullptr,FALSE);}}
HWND control(State& s,const wchar_t* type,const wchar_t* text,int id,int x,int y,int w,int h,DWORD style=0) {
    if(std::wstring_view(type)==L"STATIC")style|=SS_NOPREFIX;
    HWND child=CreateWindowExW(0,type,text,WS_CHILD|WS_VISIBLE|style,x,y,w,h,s.window,reinterpret_cast<HMENU>(INT_PTR(id)),GetModuleHandleW(nullptr),nullptr);
    SendMessageW(child,WM_SETFONT,reinterpret_cast<WPARAM>(s.font),TRUE);return child;
}
LRESULT CALLBACK procedure(HWND window,UINT message,WPARAM wp,LPARAM lp) {
    auto* s=reinterpret_cast<State*>(GetWindowLongPtrW(window,GWLP_USERDATA));
    if(message==WM_NCCREATE){s=static_cast<State*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);s->window=window;SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(s));}
    if(!s)return DefWindowProcW(window,message,wp,lp);
    try {switch(message) {
    case WM_CREATE: {
        s->font=CreateFontW(-17,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,0,0,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Segoe UI");
        control(*s,L"STATIC",L"ObjectKEY | Same object, similar view",0,24,16,900,28);
        control(*s,L"STATIC",L"Load a photo, drag a box around the complete object, then extract or compare.",0,24,50,900,26);
        s->settingsLabel=control(*s,L"STATIC",L"",0,24,88,735,28);
        control(*s,L"BUTTON",L"Reload Settings",ReloadSettings,780,84,150,32,WS_TABSTOP);showSettings(*s);
        control(*s,L"BUTTON",L"Load Image",LoadReference,24,132,104,36,WS_TABSTOP);
        control(*s,L"BUTTON",L"Extract",Extract,138,132,86,36,WS_TABSTOP);
        control(*s,L"BUTTON",L"Save",SaveTemplate,234,132,80,36,WS_TABSTOP);
        control(*s,L"BUTTON",L"Load Template",LoadTemplate,324,132,140,36,WS_TABSTOP);
        control(*s,L"BUTTON",L"Load Image",LoadQuery,490,132,140,36,WS_TABSTOP);
        control(*s,L"BUTTON",L"Compare",Compare,640,132,140,36,WS_TABSTOP);
        s->labels[0]=control(*s,L"STATIC",L"Reference image",0,24,184,440,26,SS_PATHELLIPSIS);
        s->labels[1]=control(*s,L"STATIC",L"Query image",0,490,184,440,26,SS_PATHELLIPSIS);
        for(int i=0;i<2;i++) {
            int x=panes[i].x;s->roiLabels[i]=control(*s,L"STATIC",L"Load an image to select its ROI.",0,x,554,440,26);
            control(*s,L"BUTTON",L"Full Image",i?QueryFull:ReferenceFull,x,590,130,32,WS_TABSTOP);
            control(*s,L"BUTTON",L"Edit Mask",i?QueryEdit:ReferenceEdit,x+144,590,140,32,WS_TABSTOP);
            control(*s,L"BUTTON",L"Load Mask",i?QueryMask:ReferenceMask,x+298,590,142,32,WS_TABSTOP);
        }
        s->result=control(*s,L"STATIC",L"Load a reference image or an existing template to begin.",0,24,642,906,44);
        s->status=control(*s,L"STATIC",L"Drag to select an ROI. Full Image resets it. Thresholds are read from config/matching.ini.",0,24,698,906,94);
        updateControls(*s);SetTimer(window,1,100,nullptr);return 0;
    }
    case WM_COMMAND:
        if(HIWORD(wp)!=BN_CLICKED||s->busy)return 0;
        switch(LOWORD(wp)) {
        case LoadReference:loadPhoto(*s,0);break;
        case LoadQuery:loadPhoto(*s,1);break;
        case Extract:begin(*s,true);break;
        case SaveTemplate: {
            if(!s->reference)throw std::runtime_error("Extract or load a reference template before saving.");
            auto path=choose(window,true,L"Save Extracted Template - Choose a New Filename",L"*.okt");if(!path)break;
            if(std::filesystem::exists(*path))throw std::runtime_error("Template already exists. Choose a new filename.");
            saveEmbedding(*path,*s->reference);s->templatePath=*path;
            SetWindowTextW(s->status,(L"Template saved: "+path->wstring()+L"\r\nThe current reference remains ready for comparison.").c_str());break;
        }
        case Compare:begin(*s,false);break;
        case LoadTemplate: {
            auto path=choose(window,false,L"Load Reference Template",L"*.okt");if(!path)break;
            auto record=loadEmbedding(*path);RecommendedPipeline::requireCurrentTemplate(record);
            s->templateMask=maskImage(record.local);s->reference=std::move(record);s->templatePath=*path;s->inputs[0]={};
            SetWindowTextW(s->labels[0],(L"Template: "+path->filename().wstring()).c_str());showRoi(*s,0);
            SetWindowTextW(s->result,L"Reference template loaded. Load a query image, select its ROI, then Compare.");
            SetWindowTextW(s->status,L"The saved reference mask is shown. Load the original reference image to select a new ROI.");
            updateControls(*s);InvalidateRect(window,nullptr,FALSE);break;
        }
        case ReferenceFull:case QueryFull: {
            int side=LOWORD(wp)==QueryFull?1:0;const auto& image=s->inputs[side].image;setRoi(*s,side,{0,0,image.width,image.height});break;
        }
        case ReferenceEdit:case QueryEdit:setMask(*s,LOWORD(wp)==QueryEdit?1:0,true);break;
        case ReferenceMask:case QueryMask:setMask(*s,LOWORD(wp)==QueryMask?1:0,false);break;
        case ReloadSettings: {
            auto config=loadMatchingConfig(s->config.path);s->config=std::move(config);showSettings(*s);
            SetWindowTextW(s->result,L"Settings reloaded. Click Compare to update the result.");
            SetWindowTextW(s->status,(L"Loaded settings: "+s->config.path.wstring()).c_str());break;
        }
        }return 0;
    case WM_LBUTTONDOWN:
        if(s->busy)return 0;
        for(int i=0;i<2;i++) {
            const auto& image=s->inputs[i].image;
            if(gui::contains(gui::fitImage(image.width,image.height,panes[i]),GET_X_LPARAM(lp),GET_Y_LPARAM(lp))) {
                s->dragging=i;s->anchor=s->cursor={GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};SetFocus(window);SetCapture(window);break;
            }
        }return 0;
    case WM_MOUSEMOVE:
        if(s->dragging>=0){s->cursor={GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};InvalidateRect(window,nullptr,FALSE);}return 0;
    case WM_LBUTTONUP:
        if(s->dragging>=0) {
            int side=s->dragging;const auto& image=s->inputs[side].image;
            auto roi=gui::dragRoi(image.width,image.height,gui::fitImage(image.width,image.height,panes[side]),s->anchor.x,s->anchor.y,GET_X_LPARAM(lp),GET_Y_LPARAM(lp));
            cancelDrag(*s);if(roi)setRoi(*s,side,*roi);else SetWindowTextW(s->status,L"Selection too small. Drag a larger box around the whole object.");
        }return 0;
    case WM_CAPTURECHANGED:s->dragging=-1;InvalidateRect(window,nullptr,FALSE);return 0;
    case WM_KEYDOWN:if(wp==VK_ESCAPE){cancelDrag(*s);return 0;}break;
    case WM_SETCURSOR:
        if(LOWORD(lp)==HTCLIENT&&!s->busy) {
            POINT p{};GetCursorPos(&p);ScreenToClient(window,&p);
            for(int i=0;i<2;i++){const auto& image=s->inputs[i].image;if(gui::contains(gui::fitImage(image.width,image.height,panes[i]),p.x,p.y)){SetCursor(LoadCursorW(nullptr,IDC_CROSS));return TRUE;}}
        }break;
    case WM_TIMER:finish(*s);return 0;
    case WM_PAINT:{PAINTSTRUCT ps;auto dc=BeginPaint(window,&ps);SelectObject(dc,s->font);paint(*s,dc);EndPaint(window,&ps);return 0;}
    case WM_CLOSE:if(s->busy){SetWindowTextW(s->status,L"Please wait for processing to finish before closing.");return 0;}DestroyWindow(window);return 0;
    case WM_DESTROY:KillTimer(window,1);DeleteObject(s->font);PostQuitMessage(0);return 0;
    }}catch(const std::exception& error){SetWindowTextW(s->status,widen(error.what()).c_str());MessageBoxW(window,widen(error.what()).c_str(),L"ObjectKEY Error",MB_OK|MB_ICONERROR);}
    return DefWindowProcW(window,message,wp,lp);
}
}
int runGui() {
    const HRESULT com=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);if(FAILED(com))throw std::runtime_error("Cannot initialize desktop UI.");
    struct Cleanup {~Cleanup(){CoUninitialize();}} cleanup;
    State state;state.model=locateModel();state.config=loadMatchingConfig();
    WNDCLASSW cls{};cls.lpfnWndProc=procedure;cls.hInstance=GetModuleHandleW(nullptr);cls.lpszClassName=L"ObjectKEYDesktop";
    cls.hCursor=LoadCursorW(nullptr,IDC_ARROW);cls.hbrBackground=reinterpret_cast<HBRUSH>(COLOR_WINDOW+1);RegisterClassW(&cls);
    RECT size{0,0,960,810};const DWORD style=WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX;AdjustWindowRect(&size,style,FALSE);
    HWND window=CreateWindowExW(0,cls.lpszClassName,L"ObjectKEY - Image Registration and Comparison",style,CW_USEDEFAULT,CW_USEDEFAULT,size.right-size.left,size.bottom-size.top,nullptr,nullptr,cls.hInstance,&state);
    if(!window)throw std::runtime_error("Cannot create desktop window.");ShowWindow(window,SW_SHOW);UpdateWindow(window);
    MSG message{};int status;
    while((status=GetMessageW(&message,nullptr,0,0))>0) {
        if(message.message==WM_KEYDOWN&&message.wParam==VK_ESCAPE&&state.dragging>=0){cancelDrag(state);continue;}
        if(!IsDialogMessageW(window,&message)){TranslateMessage(&message);DispatchMessageW(&message);}
    }
    return status<0?2:0;
}
}
