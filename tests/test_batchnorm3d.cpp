#include <iostream>
#include <cmath>
#include <cassert>
#include "net/Tensor3D.h"
#include "layers/BatchNorm3D.h"

int main(){
    std::cout << "=== Тест BatchNorm3D ===\n";

    // momentum=1.0f, чтобы running_mean==batch_mean, running_var==batch_var
    BatchNorm3D bn(1, 1e-5f, 1.0f);

    // Вход: все 2.0
    Tensor3D x(1,2,2,1);
    x.fill(2.0f);
    auto y = bn.forward(x, true);

    // Forward должен выдать нули
    for(int h=0; h<2; ++h){
        for(int w=0; w<2; ++w){
            assert(std::abs(y(0,h,w,0) - 0.0f) < 1e-6f);
        }
    }

    // Проверяем running_ == batch_
    float rm = bn.runningMean()[0], rv = bn.runningVar()[0];
    std::cout << "running_mean=" << rm << ", running_var=" << rv << "\n";
    assert(std::abs(rm - 2.0f) < 1e-6f);
    assert(std::abs(rv - 0.0f) < 1e-6f);

    // Backward: grad_y = 1
    Tensor3D grad_y(1,2,2,1);
    grad_y.fill(1.0f);
    auto grad_x = bn.backward(grad_y);

    // grad_gamma = Σ(dy * x̂) = 0;  grad_beta = Σ(dy) = 4
    assert(std::abs(bn.grad_gamma()[0] - 0.0f) < 1e-6f);
    assert(std::abs(bn.grad_beta()[0]  - 4.0f) < 1e-6f);

    // grad_x всё ещё нулевой
    for(int h=0; h<2; ++h){
        for(int w=0; w<2; ++w){
            assert(std::abs(grad_x(0,h,w,0) - 0.0f) < 1e-6f);
        }
    }

    std::cout << "[OK] BatchNorm3D tests passed\n";
    return 0;
}
