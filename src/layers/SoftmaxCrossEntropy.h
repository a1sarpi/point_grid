#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

/**
 * Класс для вычисления потерь Softmax + Cross-Entropy для батча.
 * При вызове forward сохраняет вероятности и метки для backward.
 */
class SoftmaxCrossEntropy {
public:
    SoftmaxCrossEntropy() = default;

    /**
     * Прямой проход.
     * @param logits  Вектор «сырых» выходов сети, размерности N*C (упакованы построчно).
     * @param labels  Вектор истинных меток длины N, значения в [0, C).
     * @return        Средняя по батчу скалярная потеря.
     */
    float forward(const std::vector<float>& logits,
                  const std::vector<int>& labels);

    /**
     * Обратный проход.
     * @return  Вектор градиентов dL/dlogits того же размера N*C.
     */
    std::vector<float> backward();

private:
    int N_ = 0;                       // размер батча
    int C_ = 0;                       // число классов
    std::vector<float> probs_;        // сохранённые softmax-вероятности (N*C)
    std::vector<int> labels_;         // сохранённые метки (N)
};
