#pragma once
#include "core/Types.h"
#include <algorithm>
#include <cmath>
#include <optional>
namespace objectkey::gui {
struct DisplayRect { int x=0,y=0,width=0,height=0; };
inline DisplayRect fitImage(unsigned width,unsigned height,DisplayRect box) {
    if(!width||!height||box.width<=0||box.height<=0)return {};
    const double scale=std::min(double(box.width)/width,double(box.height)/height);
    int w=std::max(1,int(width*scale)),h=std::max(1,int(height*scale));
    return {box.x+(box.width-w)/2,box.y+(box.height-h)/2,w,h};
}
inline bool contains(DisplayRect box,int x,int y) {
    return box.width>0&&box.height>0&&x>=box.x&&y>=box.y&&x<box.x+box.width&&y<box.y+box.height;
}
// Display coordinates are clamped to the actual image, excluding letterbox margins.
inline std::optional<Roi> dragRoi(unsigned width,unsigned height,DisplayRect display,int ax,int ay,int bx,int by) {
    if(!width||!height||display.width<=0||display.height<=0)return std::nullopt;
    ax=std::clamp(ax,display.x,display.x+display.width);bx=std::clamp(bx,display.x,display.x+display.width);
    ay=std::clamp(ay,display.y,display.y+display.height);by=std::clamp(by,display.y,display.y+display.height);
    if(std::abs(ax-bx)<4||std::abs(ay-by)<4)return std::nullopt;
    unsigned left=unsigned(std::floor(double(std::min(ax,bx)-display.x)*width/display.width));
    unsigned top=unsigned(std::floor(double(std::min(ay,by)-display.y)*height/display.height));
    unsigned right=std::min(width,unsigned(std::ceil(double(std::max(ax,bx)-display.x)*width/display.width)));
    unsigned bottom=std::min(height,unsigned(std::ceil(double(std::max(ay,by)-display.y)*height/display.height)));
    if(right-left<8||bottom-top<8)return std::nullopt;
    return Roi{left,top,right-left,bottom-top};
}
}
