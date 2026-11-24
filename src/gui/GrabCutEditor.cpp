#include "gui/GrabCutEditor.h"
#include <windowsx.h>
#include "vision/GrabCut.h"
#include <algorithm>
#include <future>
#include <chrono>
#include <stdexcept>

namespace objectkey
{
namespace
{
constexpr int Box = 201, Foreground = 202, Background = 203, Refine = 204, Reset = 205, Use = 206,
              Cancel = 207;
struct Segmentation
{
    vision::Mask labels;
};
struct Pixel
{
    int x = 0, y = 0;
};
struct Editor
{
    HWND window{}, status{};
    HFONT font{};
    std::vector<HWND> controls;
    Image image;
    vision::Mask labels, strokes;
    vision::Rect box;
    Pixel anchor, last;
    int mode = Box;
    bool drawing = false, initialized = false, dirty = true, busy = false, accepted = false;
    std::future<Segmentation> task;
    RECT display{};
};
void message(Editor &e, const wchar_t *text)
{
    SetWindowTextW(e.status, text);
}
void enable(Editor &e)
{
    for (auto h : e.controls)
        EnableWindow(h, !e.busy);
    EnableWindow(GetDlgItem(e.window, Use), !e.busy && e.initialized && !e.dirty);
}
Pixel point(Editor &e, LPARAM lp)
{
    return {std::clamp<int>(GET_X_LPARAM(lp) - e.display.left, 0, int(e.image.width) - 1),
            std::clamp<int>(GET_Y_LPARAM(lp) - e.display.top, 0, int(e.image.height) - 1)};
}
bool inside(Editor &e, LPARAM lp)
{
    POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
    return PtInRect(&e.display, p) != FALSE;
}
void stroke(Editor &e, Pixel p, bool first)
{
    unsigned char value = e.mode == Foreground ? vision::Foreground : vision::Background;
    if (first)
    {
        vision::paintDisk(e.labels, p.x, p.y, 6, value);
        vision::paintDisk(e.strokes, p.x, p.y, 6, e.mode == Foreground ? 1 : 2);
    }
    else
    {
        vision::paintLine(e.labels, e.last.x, e.last.y, p.x, p.y, 6, value);
        vision::paintLine(e.strokes, e.last.x, e.last.y, p.x, p.y, 6, e.mode == Foreground ? 1 : 2);
    }
    e.last = p;
    e.dirty = true;
    enable(e);
}
void run(Editor &e)
{
    if (e.box.width < 8 || e.box.height < 8)
        throw std::runtime_error("Draw a box enclosing the complete object, with background outside it.");
    auto image = e.image;
    auto labels = e.labels;
    auto box = e.box;
    bool initialized = e.initialized;
    // Preserve explicit strokes on refinement; a new box starts a fresh segmentation.
    e.task = std::async(std::launch::async, [image, labels, box, initialized]() mutable {
        if (!initialized)
            labels = vision::rectangleLabels(int(image.width), int(image.height), box);
        labels = vision::grabCut(image, std::move(labels), 5);
        if (vision::foregroundCount(vision::foregroundMask(labels)) < 64)
            throw std::runtime_error("No usable foreground. Draw a new box and try again.");
        return Segmentation{std::move(labels)};
    });
    e.busy = true;
    enable(e);
    message(e, L"Segmenting... Please wait.");
}
void paint(Editor &e, HDC dc)
{
    Image view = e.image;
    for (int y = 0; y < int(view.height); y++)
        for (int x = 0; x < int(view.width); x++)
        {
            auto *p = &view.pixels[(size_t(y) * view.width + x) * 3];
            auto label = e.labels.at(x, y);
            if (e.initialized && label != vision::Foreground && label != vision::ProbableForeground)
                for (int c = 0; c < 3; c++)
                    p[c] = static_cast<unsigned char>(p[c] * .22);
            auto s = e.strokes.at(x, y);
            if (s)
            {
                p[0] = s == 1 ? 40 : 30;
                p[1] = s == 1 ? 220 : 30;
                p[2] = s == 1 ? 40 : 245;
            }
            if (e.box.width > 0 && e.box.height > 0 && e.box.contains(x, y) &&
                (x < e.box.x + 2 || x >= e.box.x + e.box.width - 2 || y < e.box.y + 2 ||
                 y >= e.box.y + e.box.height - 2))
            {
                p[0] = 0;
                p[1] = 220;
                p[2] = 255;
            }
        }
    std::vector<unsigned char> bgra(size_t(view.width) * view.height * 4);
    for (size_t i = 0; i < size_t(view.width) * view.height; i++)
    {
        for (int c = 0; c < 3; c++)
            bgra[4 * i + c] = view.pixels[3 * i + c];
        bgra[4 * i + 3] = 255;
    }
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = int(view.width);
    info.bmiHeader.biHeight = -int(view.height);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    SetDIBitsToDevice(dc, e.display.left, e.display.top, view.width, view.height, 0, 0, 0, view.height,
                      bgra.data(), &info, DIB_RGB_COLORS);
}
LRESULT CALLBACK proc(HWND w, UINT msg, WPARAM wp, LPARAM lp)
{
    auto *e = reinterpret_cast<Editor *>(GetWindowLongPtrW(w, GWLP_USERDATA));
    if (msg == WM_NCCREATE)
    {
        e = static_cast<Editor *>(reinterpret_cast<CREATESTRUCTW *>(lp)->lpCreateParams);
        e->window = w;
        SetWindowLongPtrW(w, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(e));
    }
    if (!e)
        return DefWindowProcW(w, msg, wp, lp);
    try
    {
        switch (msg)
        {
        case WM_CREATE: {
            e->font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, 0, 0,
                                  CLEARTYPE_QUALITY, 0, L"Segoe UI");
            const wchar_t *names[] = {L"Draw box", L"Foreground", L"Background", L"Run / Refine",
                                      L"Reset",    L"Use mask",   L"Cancel"};
            for (int i = 0; i < 7; i++)
            {
                auto h =
                    CreateWindowW(L"BUTTON", names[i], WS_CHILD | WS_VISIBLE | WS_TABSTOP, 12 + i * 112, 12,
                                  106, 32, w, reinterpret_cast<HMENU>(INT_PTR(201 + i)), nullptr, nullptr);
                SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(e->font), TRUE);
                e->controls.push_back(h);
            }
            e->status = CreateWindowW(
                L"STATIC", L"Drag a box around the object, then Run. Green = foreground; red = background.",
                WS_CHILD | WS_VISIBLE, 12, 52, 790, 44, w, nullptr, nullptr, nullptr);
            SendMessageW(e->status, WM_SETFONT, reinterpret_cast<WPARAM>(e->font), TRUE);
            enable(*e);
            SetTimer(w, 1, 60, nullptr);
            return 0;
        }
        case WM_COMMAND:
            if (e->busy)
                return 0;
            switch (LOWORD(wp))
            {
            case Box:
                e->mode = Box;
                message(*e, L"Drag a new enclosing box. A new box clears the previous mask and corrections.");
                break;
            case Foreground:
            case Background:
                if (!e->initialized)
                {
                    message(*e, L"Draw a box and run GrabCut before painting corrections.");
                    break;
                }
                e->mode = LOWORD(wp);
                message(*e, e->mode == Foreground ? L"Paint missing object areas in green, then Run / Refine."
                                                  : L"Paint unwanted background in red, then Run / Refine.");
                break;
            case Refine:
                run(*e);
                break;
            case Reset:
                e->initialized = false;
                e->dirty = true;
                e->box = {};
                std::fill(e->labels.pixels.begin(), e->labels.pixels.end(), vision::Background);
                std::fill(e->strokes.pixels.begin(), e->strokes.pixels.end(), static_cast<unsigned char>(0));
                e->mode = Box;
                enable(*e);
                message(*e, L"Reset. Draw an enclosing box.");
                break;
            case Use:
                if (e->initialized && !e->dirty)
                {
                    e->accepted = true;
                    DestroyWindow(w);
                }
                return 0;
            case Cancel:
                DestroyWindow(w);
                return 0;
            }
            InvalidateRect(w, nullptr, FALSE);
            return 0;
        case WM_LBUTTONDOWN:
            if (e->busy || !inside(*e, lp))
                return 0;
            e->drawing = true;
            e->anchor = e->last = point(*e, lp);
            SetCapture(w);
            if (e->mode == Box)
            {
                e->initialized = false;
                e->dirty = true;
                std::fill(e->labels.pixels.begin(), e->labels.pixels.end(), vision::Background);
                std::fill(e->strokes.pixels.begin(), e->strokes.pixels.end(), static_cast<unsigned char>(0));
                e->box = {};
                enable(*e);
            }
            else
                stroke(*e, e->last, true);
            InvalidateRect(w, nullptr, FALSE);
            return 0;
        case WM_MOUSEMOVE:
        case WM_LBUTTONUP:
            if (!e->drawing || e->busy)
                return 0;
            {
                auto p = point(*e, lp);
                if (e->mode == Box)
                {
                    int x = std::max(1, std::min(p.x, e->anchor.x)),
                        y = std::max(1, std::min(p.y, e->anchor.y));
                    int right = std::min(int(e->image.width) - 1, std::max(p.x, e->anchor.x)),
                        bottom = std::min(int(e->image.height) - 1, std::max(p.y, e->anchor.y));
                    e->box = {x, y, std::max(0, right - x), std::max(0, bottom - y)};
                }
                else
                    stroke(*e, p, false);
            }
            if (msg == WM_LBUTTONUP)
            {
                e->drawing = false;
                ReleaseCapture();
            }
            InvalidateRect(w, nullptr, FALSE);
            return 0;
        case WM_TIMER:
            if (e->busy && e->task.valid() &&
                e->task.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                e->busy = false;
                auto result = e->task.get();
                e->labels = std::move(result.labels);
                e->initialized = true;
                e->dirty = false;
                std::fill(e->strokes.pixels.begin(), e->strokes.pixels.end(), static_cast<unsigned char>(0));
                enable(*e);
                message(*e, L"Inspect the mask. Paint corrections and Refine, or choose Use mask. Holes are "
                            L"preserved.");
                InvalidateRect(w, nullptr, FALSE);
            }
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            auto dc = BeginPaint(w, &ps);
            paint(*e, dc);
            EndPaint(w, &ps);
            return 0;
        }
        case WM_CAPTURECHANGED:
            e->drawing = false;
            return 0;
        case WM_CLOSE:
            if (!e->busy)
                DestroyWindow(w);
            return 0;
        case WM_DESTROY:
            KillTimer(w, 1);
            DeleteObject(e->font);
            return 0;
        }
    }
    catch (const std::exception &ex)
    {
        e->busy = false;
        e->dirty = true;
        enable(*e);
        MessageBoxA(w, ex.what(), "Mask Editor Error", MB_OK | MB_ICONERROR);
    }
    return DefWindowProcW(w, msg, wp, lp);
}
} // namespace
std::optional<Image> editGrabCutMask(HWND owner, const Image &image, const Roi &roi)
{
    if (!roi.width || !roi.height || roi.x >= image.width || roi.y >= image.height ||
        roi.width > image.width - roi.x || roi.height > image.height - roi.y ||
        image.pixels.size() != size_t(image.width) * image.height * 3)
        throw std::runtime_error("Invalid ROI for mask editing.");
    Editor e;
    double scale = std::min({1., 768. / roi.width, 570. / roi.height});
    e.image = vision::resizeArea(
        vision::crop(image, {int(roi.x), int(roi.y), int(roi.width), int(roi.height)}),
        std::max(1, int(std::lround(roi.width * scale))), std::max(1, int(std::lround(roi.height * scale))));
    if (e.image.width < 10 || e.image.height < 10)
        throw std::runtime_error("ROI is too small for GrabCut.");
    e.labels = vision::Mask(int(e.image.width), int(e.image.height), vision::Background);
    e.strokes = vision::Mask(int(e.image.width), int(e.image.height));
    e.display = {int((810 - e.image.width) / 2), 100, int((810 + e.image.width) / 2),
                 100 + int(e.image.height)};
    WNDCLASSW cls{};
    cls.lpfnWndProc = proc;
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = L"ObjectKEYMaskEditor";
    cls.hCursor = LoadCursorW(nullptr, IDC_CROSS);
    cls.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassW(&cls);
    RECT size{0, 0, 810, 680};
    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
    AdjustWindowRect(&size, style, FALSE);
    HWND w = CreateWindowExW(WS_EX_DLGMODALFRAME, cls.lpszClassName, L"GrabCut Mask Editor", style,
                             CW_USEDEFAULT, CW_USEDEFAULT, size.right - size.left, size.bottom - size.top,
                             owner, nullptr, cls.hInstance, &e);
    if (!w)
        throw std::runtime_error("Cannot create mask editor.");
    EnableWindow(owner, FALSE);
    ShowWindow(w, SW_SHOW);
    UpdateWindow(w);
    MSG msg{};
    while (IsWindow(w))
    {
        int status = GetMessageW(&msg, nullptr, 0, 0);
        if (status <= 0)
        {
            if (status == 0)
                PostQuitMessage(int(msg.wParam));
            break;
        }
        if (!IsDialogMessageW(w, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    if (IsWindow(w))
        DestroyWindow(w);
    EnableWindow(owner, TRUE);
    SetActiveWindow(owner);
    if (!e.accepted)
        return std::nullopt;
    auto full = vision::resizeNearest(vision::foregroundMask(e.labels), int(roi.width), int(roi.height));
    Image output{image.width, image.height, std::vector<unsigned char>(image.pixels.size(), 0)};
    for (unsigned y = 0; y < roi.height; y++)
        for (unsigned x = 0; x < roi.width; x++)
            for (int c = 0; c < 3; c++)
                output.pixels[(size_t(y + roi.y) * image.width + x + roi.x) * 3 + c] =
                    full.at(int(x), int(y)) ? 255 : 0;
    return output;
}
} // namespace objectkey
