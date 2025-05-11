#pragma once

#include "net/Tensor3D.h"
#include <vector>

/**
 * 3D Max-Pooling слой
 */
class MaxPool3D {
public:
    enum class Padding { VALID, SAME };

    // kD×kH×kW — окно, sD×sH×sW — страйд
    MaxPool3D(int kD, int kH, int kW,
              int sD = 1, int sH = 1, int sW = 1,
              Padding pad = Padding::VALID);

    // Прямой проход
    Tensor3D forward(const Tensor3D& x);

    // Обратный проход: принимает dL/dy, возвращает dL/dx
    Tensor3D backward(const Tensor3D& grad_y);

    // Сбросить накопленные градиенты
    void zeroGrad();

    // Геттер накопленного dL/dx (после backward)
    const Tensor3D& gradInput() const { return gradInput_; }

private:
    int kD_, kH_, kW_;
    int sD_, sH_, sW_;
    Padding pad_;

    // Форма последнего forward-входа
    int inD_, inH_, inW_, inC_;
    // Форма последнего forward-выхода
    int outD_, outH_, outW_;

    // Для каждого выходного элемента — flatten-индекс, откуда брать grad
    std::vector<int> maxIndex_;

    // Накопленные градиенты по входу
    Tensor3D gradInput_;
};
