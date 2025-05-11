#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fstream>

#include "net/Tensor3D.h"
#include "layers/Conv3D.h"
#include "layers/BatchNorm3D.h"
#include "layers/ReLU3D.h"
#include "layers/MaxPool3D.h"
#include "layers/FullyConnected.h"
#include "layers/SoftmaxCrossEntropy.h"
#include "optim/SGD.h"

/**
 * Обёртка над всей 3D-CNN + оптимизатор SGD с моментом.
 */
class Network {
public:
    Network();

    // Прямой проход: input → логиты
    std::vector<float> forward(const Tensor3D& input, bool training = true);

    // Вычислить потерю (SoftmaxCrossEntropy)
    float computeLoss(const std::vector<int>& labels);

    // Обратный проход
    void backward();

    // Шаг оптимизации
    void optimize();

    // Сброс всех градиентов в слоях и оптимизаторе
    void zeroGrad();

    // Сохранение и загрузка состояния (чекпоинт)
    void saveCheckpoint(const std::string& filepath) const;
    void loadCheckpoint(const std::string& filepath);

private:
    Conv3D              conv1_;
    BatchNorm3D         bn1_;
    ReLU3D              relu1_;
    MaxPool3D           pool1_;
    FullyConnected      fc_;
    SoftmaxCrossEntropy criterion_;
    SGD                 optimizer_;

    // Буферы для промежуточных результатов
    Tensor3D            conv_out_, bn_out_, relu_out_, pool_out_;
    std::vector<float>  fc_out_;
};
