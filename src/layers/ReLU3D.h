#pragma once
#include "net/Tensor3D.h"
#include <vector>

class ReLU3D {
public:
    ReLU3D() = default;

    // forward сохраняет маску (x>0) и возвращает y
    Tensor3D forward(const Tensor3D& x);

    // backward умножает grad_y на маску и проверяет форму
    Tensor3D backward(const Tensor3D& grad_y);

    // полный сброс состояния
    void zeroGrad();

private:
    int D_=0, H_=0, W_=0, C_=0;
    std::vector<uint8_t> mask_;  // size = D_*H_*W_*C_
};
