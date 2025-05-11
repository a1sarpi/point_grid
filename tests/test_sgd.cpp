#include "optim/SGD.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

static bool almostEqual(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) < tol;
}

int main() {
    std::cout << "=== Тест SGD ===\n";

    // Параметр и начальный градиент
    std::vector<float> w = {1.0f, -2.0f};
    std::vector<float> g = {0.5f, -1.5f};

    SGD opt(0.1f, 0.9f);
    opt.addParam(w, g);

    // Первый шаг
    opt.step();
    // v = 0*0.9 - 0.1*{0.5, -1.5} = {-0.05, +0.15}
    // w = {1.0 - 0.05, -2.0 + 0.15} = {0.95, -1.85}
    assert(almostEqual(w[0], 0.95f));
    assert(almostEqual(w[1], -1.85f));

    // Обновляем градиент и второй шаг
    g = {0.2f, 0.3f};
    opt.step();
    // v = 0.9*{-0.05,0.15} - 0.1*{0.2,0.3} = {-0.045-0.02, 0.135-0.03} = {-0.065,0.105}
    // w = {0.95-0.065, -1.85+0.105} = {0.885, -1.745}
    assert(almostEqual(w[0], 0.885f));
    assert(almostEqual(w[1], -1.745f));

    // Сброс состояния момента и третий шаг с новым градиентом
    opt.zeroState();
    g = {1.0f, 1.0f};
    opt.step();
    // v = 0 - 0.1*1 = -0.1
    // w = {0.885-0.1, -1.745-0.1} = {0.785, -1.845}
    assert(almostEqual(w[0], 0.785f));
    assert(almostEqual(w[1], -1.845f));

    std::cout << "[OK] Все проверки пройдены!\n";
    return 0;
}
