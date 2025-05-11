#include "layers/SoftmaxCrossEntropy.h"

float SoftmaxCrossEntropy::forward(const std::vector<float>& logits,
                                   const std::vector<int>& labels) {
    assert(!logits.empty());
    N_ = static_cast<int>(labels.size());
    assert(static_cast<int>(logits.size()) % N_ == 0);
    C_ = static_cast<int>(logits.size()) / N_;
    probs_.assign(logits.size(), 0.0f);
    labels_ = labels;

    float loss = 0.0f;
    // для каждой выборки i вычисляем softmax
    for (int i = 0; i < N_; ++i) {
        // найти max для стабильности
        float maxLogit = logits[i*C_];
        for (int j = 1; j < C_; ++j) {
            maxLogit = std::max(maxLogit, logits[i*C_ + j]);
        }
        // sum exp
        float sumExp = 0.0f;
        for (int j = 0; j < C_; ++j) {
            float e = std::exp(logits[i*C_ + j] - maxLogit);
            probs_[i*C_ + j] = e;
            sumExp += e;
        }
        float logSumExp = std::log(sumExp) + maxLogit;
        // нормируем
        for (int j = 0; j < C_; ++j) {
            probs_[i*C_ + j] /= sumExp;
        }
        int lbl = labels_[i];
        assert(lbl >= 0 && lbl < C_);
        loss += (logSumExp - logits[i*C_ + lbl]);  // -log p_lbl = logSumExp - logit_lbl
    }
    return loss / N_;
}

std::vector<float> SoftmaxCrossEntropy::backward() {
    assert(static_cast<int>(probs_.size()) == N_ * C_);
    // Правильная инициализация градиентов размером N_*C_
    std::vector<float> grad(N_ * C_, 0.0f);
    // dL/dlogit = (p - y) / N
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < C_; ++j) {
            float p = probs_[i*C_ + j];
            grad[i*C_ + j] = p - (j == labels_[i] ? 1.0f : 0.0f);
        }
    }
    // усредняем по батчу
    float invN = 1.0f / N_;
    for (auto& g : grad) {
        g *= invN;
    }
    return grad;
}

