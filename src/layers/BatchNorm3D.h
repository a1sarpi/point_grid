#pragma once

#include "net/Tensor3D.h"
#include <vector>
#include <cassert>

/**
 * 3D Batch Normalization по каналам.
 *
 * Формулы:
 *   μ[c] = 1/N * Σ_i x[i,c]
 *   σ²[c] = 1/N * Σ_i (x[i,c] - μ[c])²
 *   x̂[i,c] = (x[i,c] - μ[c]) / sqrt(σ²[c] + eps)
 *   y[i,c] = γ[c] * x̂[i,c] + β[c]
 *
 * backward:
 *   grad_beta[c]  = Σ_i dy[i,c]
 *   grad_gamma[c] = Σ_i dy[i,c] * x̂[i,c]
 *   dL/dx[i,c] = (γ[c] * inv_std[c] / N) *
 *       (N * dy[i,c] - grad_beta[c] - x̂[i,c] * grad_gamma[c])
 */
class BatchNorm3D {
public:
    BatchNorm3D(int channels, float eps = 1e-5f, float momentum = 0.1f);

    // Прямой проход: training=true — обновляем running_*,
    // training=false — используем running_* без сохранения batch_*
    Tensor3D forward(const Tensor3D& x, bool training);

    // Обратный проход (только после forward training=true)
    Tensor3D backward(const Tensor3D& grad_y);

    // Сбросить накопленные градиенты γ и β, а также временные буферы
    void zeroGrad();

    // Геттеры параметров и их градиентов
    const std::vector<float>& gamma()      const { return gamma_; }
    const std::vector<float>& beta()       const { return beta_; }
    const std::vector<float>& grad_gamma() const { return grad_gamma_; }
    const std::vector<float>& grad_beta()  const { return grad_beta_; }

    // Геттеры running-статистик
    const std::vector<float>& runningMean() const { return running_mean_; }
    const std::vector<float>& runningVar()  const { return running_var_; }

private:
    int C_;               // число каналов
    float eps_, momentum_;

    // learnable
    std::vector<float> gamma_, beta_;
    // их градиенты
    std::vector<float> grad_gamma_, grad_beta_;

    // скользящие статистики (для inference)
    std::vector<float> running_mean_, running_var_;

    // временные буферы, только valid после forward(training=true)
    int D_, H_, W_, N_;                       // N_ = D_*H_*W_
    std::vector<float> batch_mean_, batch_var_, inv_std_, x_hat_;
};
