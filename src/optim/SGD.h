// SGD.h
#pragma once

#include "IOptimizer.h"
#include <vector>
#include <stdexcept>

/**
 * Алгоритм стохастического градиентного спуска с моментумом.
 */
class SGD : public IOptimizer {
public:
    SGD(float learning_rate = 0.01f, float momentum = 0.0f);

    // Регистрация параметров и их градиентов
    void addParam(std::vector<float>& param,
                  std::vector<float>& grad) override;

    // Один шаг обновления: v = momentum*v - lr*grad; param += v
    void step() override;

    // Сброс градиентов
    void zeroGrad() override;

    ~SGD() override = default;

    // Сериализация / десериализация буферов момента
    std::vector<std::vector<float>> getVelocityStates() const;
    void setVelocityStates(const std::vector<std::vector<float>>& vel_states);

private:
    float lr_, momentum_;

    struct ParamState {
        std::vector<float>* param;
        std::vector<float>* grad;
        std::vector<float>  velocity;
    };

    std::vector<ParamState> states_;
};
