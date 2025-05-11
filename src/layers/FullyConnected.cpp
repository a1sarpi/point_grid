#include "layers/FullyConnected.h"

// Единый статический генератор для всех слоёв
static std::mt19937& global_gen() {
    static std::mt19937 gen{ std::random_device{}() };
    return gen;
}

FullyConnected::FullyConnected(int in_features, int out_features)
    : in_f_(in_features), out_f_(out_features),
      weight_(in_features * out_features),
      bias_(out_features, 0.0f),
      grad_weight_(in_features * out_features, 0.0f),
      grad_bias_(out_features, 0.0f)
{
    assert(in_f_ > 0 && out_f_ > 0);
    initWeightsXavier();
}

void FullyConnected::initWeightsXavier() noexcept {
    float bound = std::sqrt(6.0f / float(in_f_ + out_f_));
    std::uniform_real_distribution<float> dist(-bound, bound);
    auto& gen = global_gen();
    for (auto& w : weight_) {
        w = dist(gen);
    }
    // bias_ уже инициализированы нулями
}

std::vector<float> FullyConnected::forward(const std::vector<float>& x) {
    assert(int(x.size()) == in_f_);
    input_ = x;
    std::vector<float> y(out_f_, 0.0f);

    for (int o = 0; o < out_f_; ++o) {
        float sum = bias_[o];
        int base = o * in_f_;
        for (int i = 0; i < in_f_; ++i) {
            sum += weight_[base + i] * x[i];
        }
        y[o] = sum;
    }
    return y;
}

std::vector<float> FullyConnected::backward(const std::vector<float>& grad_y) {
    assert(int(grad_y.size()) == out_f_);
    std::vector<float> grad_x(in_f_, 0.0f);

    // grad_bias
    for (int o = 0; o < out_f_; ++o) {
        grad_bias_[o] += grad_y[o];
    }
    // grad_weight and grad_x
    for (int o = 0; o < out_f_; ++o) {
        int base = o * in_f_;
        for (int i = 0; i < in_f_; ++i) {
            grad_weight_[base + i] += input_[i] * grad_y[o];
            grad_x[i]               += weight_[base + i] * grad_y[o];
        }
    }
    return grad_x;
}

void FullyConnected::zeroGrad() {
    std::fill(grad_weight_.begin(), grad_weight_.end(), 0.0f);
    std::fill(grad_bias_.begin(),   grad_bias_.end(),   0.0f);
}
