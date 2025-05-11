#include "network.h"

Network::Network()
    : conv1_(1, 16, 3, 3, 3, 1, 1, 1, Conv3D::Padding::SAME),
      bn1_(16),
      relu1_(),
      pool1_(2, 2, 2, 2, 2, 2, MaxPool3D::Padding::VALID),
      fc_(16 * 4 * 4 * 4, 10),
      criterion_(),
      optimizer_(0.01f, 0.9f)
{
    // Регистрация параметров в оптимизаторе
    optimizer_.addParam(conv1_.weight(), conv1_.weightGrad());
    optimizer_.addParam(conv1_.bias(),   conv1_.biasGrad());
    optimizer_.addParam(bn1_.gamma(),    bn1_.grad_gamma());
    optimizer_.addParam(bn1_.beta(),     bn1_.grad_beta());
    optimizer_.addParam(fc_.weight(),    fc_.gradWeight());
    optimizer_.addParam(fc_.bias(),      fc_.gradBias());
}

std::vector<float> Network::forward(const Tensor3D& input, bool training) {
    conv_out_ = conv1_.forward(input);
    bn_out_   = bn1_.forward(conv_out_, training);
    relu_out_ = relu1_.forward(bn_out_);
    pool_out_ = pool1_.forward(relu_out_);
    fc_out_   = fc_.forward(pool_out_.flatten());
    return fc_out_;
}

float Network::computeLoss(const std::vector<int>& labels) {
    return criterion_.forward(fc_out_, labels);
}

void Network::backward() {
    auto grad_logits = criterion_.backward();
    auto grad_fc     = fc_.backward(grad_logits);

    Tensor3D grad_pool(pool_out_.depth(),
                       pool_out_.height(),
                       pool_out_.width(),
                       pool_out_.channels());
    std::copy(grad_fc.begin(), grad_fc.end(), grad_pool.data());

    auto grad_relu = pool1_.backward(grad_pool);
    auto grad_bn   = relu1_.backward(grad_relu);
    auto grad_conv = bn1_.backward(grad_bn);
    conv1_.backward(conv_out_, grad_conv);
}

void Network::optimize() {
    optimizer_.step();
}

void Network::zeroGrad() {
    conv1_.zeroGrad();
    bn1_.zeroGrad();
    relu1_.zeroGrad();
    pool1_.zeroGrad();
    fc_.zeroGrad();
    optimizer_.zeroGrad();
}

void Network::saveCheckpoint(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Не удалось открыть файл для записи чекпоинта: " + filepath);

    // Сериализуем conv1 weights и bias
    auto writeVec = [&](const std::vector<float>& v) {
        int n = static_cast<int>(v.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * n);
    };
    writeVec(conv1_.weight());
    writeVec(conv1_.bias());

    // Сериализуем BatchNorm γ и β
    writeVec(bn1_.gamma());
    writeVec(bn1_.beta());

    // Сериализуем FullyConnected weights и bias
    writeVec(fc_.weight());
    writeVec(fc_.bias());

    // Сериализуем состояние оптимизатора (velocity buffers)
    auto vels = optimizer_.getVelocityStates();
    int m = static_cast<int>(vels.size());
    out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    for (const auto& vel : vels) {
        writeVec(vel);
    }
}

void Network::loadCheckpoint(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Не удалось открыть файл для чтения чекпоинта: " + filepath);

    auto readVec = [&](std::vector<float>& v) {
        int n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        v.resize(n);
        in.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n);
    };
    readVec(conv1_.weight());
    readVec(conv1_.bias());

    readVec(bn1_.gamma());
    readVec(bn1_.beta());

    readVec(fc_.weight());
    readVec(fc_.bias());

    // Восстанавливаем состояние оптимизатора
    int m;
    in.read(reinterpret_cast<char*>(&m), sizeof(m));
    std::vector<std::vector<float>> vels;
    vels.reserve(m);
    for (int i = 0; i < m; ++i) {
        std::vector<float> vel;
        readVec(vel);
        vels.push_back(std::move(vel));
    }
    optimizer_.setVelocityStates(vels);
}
