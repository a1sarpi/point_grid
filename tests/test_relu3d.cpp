#include <iostream>
#include <cassert>
#include "net/Tensor3D.h"
#include "layers/ReLU3D.h"

int main(){
    std::cout << "=== Тест ReLU3D ===\n";

    // 1) Forward
    Tensor3D x(1,2,2,1);
    x(0,0,0,0) = -1.0f;  x(0,0,1,0) =  2.0f;
    x(0,1,0,0) =  0.0f;  x(0,1,1,0) = -3.0f;

    ReLU3D act;
    auto y = act.forward(x);

    std::cout << "Forward output:\n";
    std::cout << y(0,0,0,0) << " " << y(0,0,1,0) << "\n";
    std::cout << y(0,1,0,0) << " " << y(0,1,1,0) << "\n";
    // ожидаем [0, 2; 0, 0]
    assert(y(0,0,0,0) == 0.0f);
    assert(y(0,0,1,0) == 2.0f);
    assert(y(0,1,0,0) == 0.0f);
    assert(y(0,1,1,0) == 0.0f);

    // 2) Backward: grad_y = единицы
    Tensor3D grad_y(1,2,2,1);
    grad_y.fill(1.0f);
    auto grad_x = act.backward(grad_y);

    std::cout << "Backward gradInput:\n";
    std::cout << grad_x(0,0,0,0) << " " << grad_x(0,0,1,0) << "\n";
    std::cout << grad_x(0,1,0,0) << " " << grad_x(0,1,1,0) << "\n";
    // ожидаем [0,1;0,0]
    assert(grad_x(0,0,0,0) == 0.0f);
    assert(grad_x(0,0,1,0) == 1.0f);
    assert(grad_x(0,1,0,0) == 0.0f);
    assert(grad_x(0,1,1,0) == 0.0f);

    std::cout << "[OK] ReLU3D all tests passed\n";
    return 0;
}
