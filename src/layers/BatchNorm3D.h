#pragma once

#include "net/Tensor3D.h"
#include <vector>
#include <cassert>

/**
 * 3D-BatchNorm по каналам.
 */
class BatchNorm3D {
public:
    BatchNorm3D(int channels, float eps = 1e-5f, float momentum = 0.1f);

    Tensor3D forward(const Tensor3D& x, bool training);
    Tensor3D backward(const Tensor3D& grad_y);
    void zeroGrad();

    // НЕКОНСТАНТНЫЕ геттеры для оптимизатора
    std::vector<float>& gamma()      { return gamma_; }
    std::vector<float>& beta()       { return beta_; }
    std::vector<float>& grad_gamma() { return grad_gamma_; }
    std::vector<float>& grad_beta()  { return grad_beta_; }

    // Константные геттеры
    const std::vector<float>& gamma()      const { return gamma_; }
    const std::vector<float>& beta()       const { return beta_; }
    const std::vector<float>& grad_gamma() const { return grad_gamma_; }
    const std::vector<float>& grad_beta()  const { return grad_beta_; }

    // для inference
    const std::vector<float>& runningMean() const { return running_mean_; }
    const std::vector<float>& runningVar()  const { return running_var_; }

private:
    int C_;
    float eps_, momentum_;

    std::vector<float> gamma_, beta_;
    std::vector<float> grad_gamma_, grad_beta_;
    std::vector<float> running_mean_, running_var_;

    // временные буферы для backward
    int D_, H_, W_, N_;
    std::vector<float> batch_mean_, batch_var_, inv_std_, x_hat_;
};
