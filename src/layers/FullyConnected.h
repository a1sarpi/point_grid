#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cassert>

/**
 * Полносвязный (Dense) слой
 * y = W * x + b
 */
class FullyConnected {
public:
    // Конструктор: in_features -> out_features
    FullyConnected(int in_features, int out_features);

    // Прямой проход: сохраняет input_ и возвращает y
    std::vector<float> forward(const std::vector<float>& x);

    // Обратный проход: накапливает gradWeight_, gradBias_ и возвращает grad_x
    std::vector<float> backward(const std::vector<float>& grad_y);

    // Сброс градиентов
    void zeroGrad();

    // Геттеры параметров и градиентов
    const std::vector<float>& weight()     const { return weight_; }
    const std::vector<float>& bias()       const { return bias_; }
    const std::vector<float>& gradWeight() const { return grad_weight_; }
    const std::vector<float>& gradBias()   const { return grad_bias_; }

private:
    int in_f_, out_f_;
    std::vector<float> weight_, bias_;
    std::vector<float> grad_weight_, grad_bias_;
    std::vector<float> input_;  // сохранённый вход для backward

    // Инициализация весов по методу Ксавье
    void initWeightsXavier() noexcept;
};
