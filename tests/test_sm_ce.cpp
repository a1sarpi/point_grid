#include <iostream>
#include <cmath>
#include <cassert>
#include "layers/SoftmaxCrossEntropy.h"

static bool almostEqual(float a, float b, float tol=1e-6f) {
    return std::fabs(a - b) < tol;
}

int main(){
    std::cout << "=== Тест SoftmaxCrossEntropy ===\n";

    SoftmaxCrossEntropy crit;

    // 1) Простейший случай: N=1, C=3, logits all equal
    {
        std::vector<float> logits = {0.0f, 0.0f, 0.0f};
        std::vector<int>   labels = {1};
        float loss = crit.forward(logits, labels);
        // softmax = [1/3,1/3,1/3], loss = -log(1/3) = ln(3)
        float expected_loss = std::log(3.0f);
        std::cout << "Loss = " << loss << " (ожидаем " << expected_loss << ")\n";
        assert(almostEqual(loss, expected_loss));

        auto grad = crit.backward();
        // grad = [1/3,1/3-1,1/3] = [0.3333, -0.6667, 0.3333]
        std::vector<float> expected_grad = {1.0f/3, -2.0f/3, 1.0f/3};
        for(int j=0;j<3;++j){
            std::cout << "grad["<<j<<"] = "<<grad[j]
                      << " (ожидаем "<<expected_grad[j]<<")\n";
            assert(almostEqual(grad[j], expected_grad[j]));
        }
    }

    // 2) Два примера, C=2
    {
        // пример 0: logits [2,0], label=0 → softmax=[e2/(e2+1),1/(e2+1)]
        // пример 1: logits [0,1], label=1 → softmax=[1/(1+e1), e1/(1+e1)]
        std::vector<float> logits = {2.0f, 0.0f,   0.0f, 1.0f};
        std::vector<int>   labels = {0, 1};
        float loss = crit.forward(logits, labels);
        // loss = (−log p0 + −log p1)/2
        float p00 = std::exp(2.0f)/(std::exp(2.0f)+1.0f);
        float p11 = std::exp(1.0f)/(1.0f+std::exp(1.0f));
        float expected_loss = ( -std::log(p00) -std::log(p11) ) / 2.0f;
        std::cout << "Loss2 = " << loss << " (ожидаем "<<expected_loss<<")\n";
        assert(almostEqual(loss, expected_loss));

        auto grad = crit.backward();
        // grad[0] = p00 - 1, grad[1] = p01
        // grad[2] = p10,   grad[3] = p11 - 1
        std::vector<float> expected_grad = {
            p00 - 1.0f, (1.0f-p00),
            (1.0f-p11), p11 - 1.0f
        };
        for(int i=0;i<4;++i){
            std::cout<<"grad["<<i<<"] = "<<grad[i]
                     <<" (ожидаем "<<expected_grad[i]<<")\n";
            // backward divides by N so here we need to divide by 2
            assert(almostEqual(grad[i], expected_grad[i]/2.0f));
        }
    }

    std::cout << "[OK] SoftmaxCrossEntropy tests passed\n";
    return 0;
}
