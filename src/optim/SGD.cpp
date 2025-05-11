#include "SGD.h"

SGD::SGD(float learning_rate, float momentum)
    : lr_(learning_rate), momentum_(momentum)
{
    if (lr_ <= 0.0f)
        throw std::invalid_argument("SGD: learning_rate должен быть > 0");
    if (momentum_ < 0.0f || momentum_ >= 1.0f)
        throw std::invalid_argument("SGD: momentum должен быть в [0, 1)");
}

void SGD::addParam(std::vector<float>& param, std::vector<float>& grad) {
    if (param.size() != grad.size())
        throw std::invalid_argument("SGD::addParam: размеры param и grad должны совпадать");
    ParamState st;
    st.param    = &param;
    st.grad     = &grad;
    st.velocity.assign(param.size(), 0.0f);
    states_.push_back(std::move(st));
}

void SGD::step() {
    for (auto& st : states_) {
        auto& p = *st.param;
        auto& g = *st.grad;
        auto& v = st.velocity;
        for (size_t i = 0; i < p.size(); ++i) {
            v[i] = momentum_ * v[i] - lr_ * g[i];
            p[i] += v[i];
        }
    }
}

void SGD::zeroGrad() {
    for (auto& st : states_) {
        std::fill(st.grad->begin(), st.grad->end(), 0.0f);
    }
}

std::vector<std::vector<float>> SGD::getVelocityStates() const {
    std::vector<std::vector<float>> vels;
    vels.reserve(states_.size());
    for (const auto& st : states_) {
        vels.push_back(st.velocity);
    }
    return vels;
}

void SGD::setVelocityStates(const std::vector<std::vector<float>>& vel_states) {
    if (vel_states.size() != states_.size())
        throw std::runtime_error("SGD::setVelocityStates: количество буферов не совпадает");
    for (size_t i = 0; i < states_.size(); ++i) {
        if (vel_states[i].size() != states_[i].velocity.size())
            throw std::runtime_error("SGD::setVelocityStates: размер внутреннего буфера не совпадает");
        states_[i].velocity = vel_states[i];
    }
}
