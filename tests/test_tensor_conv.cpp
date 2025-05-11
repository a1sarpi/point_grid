#include <iostream>
#include <cassert>
#include "net/Tensor3D.h"
#include "layers/Conv3D.h"

int main(){
    // ===== 1. Проверка Tensor3D =====
    std::cout << "=== Тест Tensor3D ===\n";
    Tensor3D t(2,2,2,1);
    // Заполняем: t(d,h,w,0) = d*4 + h*2 + w
    for(int d=0; d<2; ++d)
    for(int h=0; h<2; ++h)
    for(int w=0; w<2; ++w){
        t(d,h,w,0) = float(d*4 + h*2 + w);
    }
    // Проверяем shape
    std::cout << "Depth=" << t.depth() << ", Height=" << t.height()
              << ", Width=" << t.width() << ", Channels=" << t.channels() << "\n";
    assert(t.depth()==2 && t.height()==2 && t.width()==2 && t.channels()==1);

    // flatten
    auto flat = t.flatten();
    std::cout << "Flatten: ";
    for(auto v: flat) std::cout << v << " ";
    std::cout << "\n";
    assert((int)flat.size()==8);
    for(int i=0;i<8;++i) {
        std::cout << "flat["<<i<<"] = " << flat[i]
                  << " (expected " << i << ")\n";
        assert(flat[i] == float(i));
    }

    // reshape и проверка
    t.reshape(4,1,2,1);
    std::cout << "После reshape: Depth=" << t.depth()
              << ", Height=" << t.height()
              << ", Width=" << t.width()
              << ", Channels=" << t.channels() << "\n";
    assert(t.depth()==4 && t.height()==1 && t.width()==2 && t.channels()==1);

    std::cout<<"[OK] Tensor3D all tests passed\n\n";

    // ===== 2. Проверка Conv3D (forward) =====
    std::cout << "=== Тест Conv3D forward ===\n";
    // Вход: 1×3×3×1, значения 1..9
    Tensor3D in(1,3,3,1);
    int v=1;
    for(int h=0;h<3;++h)
    for(int w=0;w<3;++w){
        in(0,h,w,0) = float(v++);
    }
    std::cout << "Input:\n";
    for(int h=0;h<3;++h){
        for(int w=0;w<3;++w){
            std::cout << in(0,h,w,0) << " ";
        }
        std::cout << "\n";
    }

    // Конфигурация свёртки
    Conv3D conv(1,1,1,2,2,1,1,1, Conv3D::Padding::VALID);
    // Все веса =1, bias=0
    {
        auto& W = const_cast<std::vector<float>&>(conv.weight());
        std::fill(W.begin(), W.end(), 1.0f);
        auto& B = const_cast<std::vector<float>&>(conv.bias());
        std::fill(B.begin(), B.end(), 0.0f);
    }

    // Выполняем forward
    auto out = conv.forward(in);
    std::cout << "Output shape: "
              << out.depth() << "×"
              << out.height() << "×"
              << out.width() << "×"
              << out.channels() << "\n";
    assert(out.depth()==1 && out.height()==2 && out.width()==2 && out.channels()==1);

    // Ожидаемые значения
    float expected[2][2] = {
        { 1+2+4+5,  2+3+5+6},
        { 4+5+7+8,  5+6+8+9}
    };

    std::cout << "Output values vs expected:\n";
    for(int i=0;i<2;++i){
        for(int j=0;j<2;++j){
            float actual = out(0,i,j,0);
            float expv   = expected[i][j];
            std::cout << "out(0,"<<i<<","<<j<<",0) = "
                      << actual << " (expected " << expv << ")\n";
            assert(std::abs(actual - expv) < 1e-6f);
        }
    }

    std::cout<<"[OK] Conv3D forward all tests passed\n";
    return 0;
}
