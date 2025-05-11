#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cassert>

/**
 * Полносвязный (Dense) слой: y = W * x + b
 */
class FullyConnected {
public:
    FullyConnected(int in_features, int out_features);

    std::vector<float> forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& grad_y);
    void zeroGrad();

    // НЕКОНСТАНТНЫЕ геттеры для оптимизатора
    std::vector<float>& weight()     { return weight_; }
    std::vector<float>& bias()       { return bias_; }
    std::vector<float>& gradWeight() { return grad_weight_; }
    std::vector<float>& gradBias()   { return grad_bias_; }

    // Константные геттеры
    const std::vector<float>& weight()     const { return weight_; }
    const std::vector<float>& bias()       const { return bias_; }
    const std::vector<float>& gradWeight() const { return grad_weight_; }
    const std::vector<float>& gradBias()   const { return grad_bias_; }

private:
    int in_f_, out_f_;
    std::vector<float> weight_, bias_;
    std::vector<float> grad_weight_, grad_bias_;
    std::vector<float> input_;  // сохранённый вход для backward

    void initWeightsXavier() noexcept;
};
