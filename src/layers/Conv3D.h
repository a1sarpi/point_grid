#pragma once

#include "net/Tensor3D.h"
#include <vector>

/**
 * 3D-свёртка:
 *   на вход Tensor3D (D×H×W×in_ch)
 *   возвращает Tensor3D (D_out×H_out×W_out×out_ch)
 */
class Conv3D {
public:
    enum class Padding { VALID, SAME };

    Conv3D(int in_ch, int out_ch,
           int kD, int kH, int kW,
           int sD=1, int sH=1, int sW=1,
           Padding pad=Padding::SAME);

    // Инициализация весов (Xavier)
    void initWeightsXavier();

    // Прямой проход
    Tensor3D forward(const Tensor3D& x);

    // Обратный проход: на вход gradients по выходу,
    // возвращает gradients по входу
    Tensor3D backward(const Tensor3D& x, const Tensor3D& grad_out);

    // Сбросить накопленные градиенты
    void zeroGrad();

    // Доступ к параметрам и градиентам
    const std::vector<float>& weight()     const { return weight_; }
    const std::vector<float>& bias()       const { return bias_; }
    const std::vector<float>& weightGrad() const { return grad_w_; }
    const std::vector<float>& biasGrad()   const { return grad_b_; }

private:
    int in_ch_, out_ch_;
    int kD_, kH_, kW_;
    int sD_, sH_, sW_;
    Padding pad_;

    std::vector<float> weight_;  // size = kD*kH*kW*in_ch*out_ch
    std::vector<float> bias_;    // size = out_ch
    std::vector<float> grad_w_;  // такого же размера
    std::vector<float> grad_b_;  // size = out_ch

    // Индекс в буфере weight_
    inline int wIndex(int od, int oh, int ow, int ic, int oc) const {
        int idx = (((od * kH_ + oh) * kW_ + ow)
                   * in_ch_ + ic) * out_ch_ + oc;
        return idx;
    }

    void computeOutputDims(int D, int H, int W,
                           int& D_out, int& H_out, int& W_out) const;
};
