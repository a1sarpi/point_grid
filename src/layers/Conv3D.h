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
           int sD = 1, int sH = 1, int sW = 1,
           Padding pad = Padding::SAME);

    void initWeightsXavier();
    Tensor3D forward(const Tensor3D& x);
    Tensor3D backward(const Tensor3D& x, const Tensor3D& grad_out);
    void zeroGrad();

    // НЕКОНСТАНТНЫЕ геттеры для оптимизатора
    std::vector<float>& weight()     { return weight_; }
    std::vector<float>& bias()       { return bias_; }
    std::vector<float>& weightGrad() { return grad_w_; }
    std::vector<float>& biasGrad()   { return grad_b_; }

    // Константные геттеры (если вам нужно читать параметры)
    const std::vector<float>& weight()     const { return weight_; }
    const std::vector<float>& bias()       const { return bias_; }
    const std::vector<float>& weightGrad() const { return grad_w_; }
    const std::vector<float>& biasGrad()   const { return grad_b_; }

private:
    int in_ch_, out_ch_;
    int kD_, kH_, kW_;
    int sD_, sH_, sW_;
    Padding pad_;

    std::vector<float> weight_;    // size = kD*kH*kW*in_ch*out_ch
    std::vector<float> bias_;      // size = out_ch
    std::vector<float> grad_w_;    // того же размера, что weight_
    std::vector<float> grad_b_;    // size = out_ch

    inline int wIndex(int od, int oh, int ow, int ic, int oc) const {
        return (((od * kH_ + oh) * kW_ + ow)
                * in_ch_ + ic) * out_ch_ + oc;
    }

    void computeOutputDims(int D, int H, int W,
                           int& D_out, int& H_out, int& W_out) const;
};
