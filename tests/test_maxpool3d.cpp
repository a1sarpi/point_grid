#include <iostream>
#include <cassert>
#include "net/Tensor3D.h"
#include "layers/MaxPool3D.h"

int main(){
    std::cout << "=== Тест MaxPool3D ===\n";

    Tensor3D t(1,2,2,1);
    // [1 2]
    // [3 4]
    t(0,0,0,0)=1; t(0,0,1,0)=2;
    t(0,1,0,0)=3; t(0,1,1,0)=4;

    MaxPool3D mp(1,2,2,1,1,1, MaxPool3D::Padding::VALID);
    auto y = mp.forward(t);

    std::cout << "Output shape: "
              << y.depth() << "×" << y.height() << "×"
              << y.width() << "×" << y.channels() << "\n";
    std::cout << "Output value: " << y(0,0,0,0)
              << " (expected 4)\n";
    assert(y(0,0,0,0) == 4.0f);

    Tensor3D grad_y(1,1,1,1);
    grad_y(0,0,0,0) = 1.0f;
    auto grad_in = mp.backward(grad_y);

    std::cout << "GradInput flattened: [";
    for(int i=0;i<4;i++){
      std::cout << grad_in.data()[i]
                << (i<3 ? ", " : "");
    }
    std::cout << "] (expected [0, 0, 0, 1])\n";
    assert(grad_in.data()[3] == 1.0f);
    for(int i=0;i<3;i++) assert(grad_in.data()[i] == 0.0f);

    std::cout << "[OK] MaxPool3D tests passed\n";
    return 0;
}
