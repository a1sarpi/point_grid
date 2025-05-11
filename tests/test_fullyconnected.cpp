#include <iostream>
#include <cassert>
#include "layers/FullyConnected.h"

int main(){
    std::cout << "=== Тест FullyConnected ===\n";

    // 1) Создадим слой 2→2 и вручную зададим W и b
    FullyConnected fc(2, 2);
    {
        // Поменяем метод init, захардкодим
        auto& W = const_cast<std::vector<float>&>(fc.weight());
        // Формат: row-major: [ W00, W01,  W10, W11 ]
        W = { 1.0f, 0.0f,
              0.0f, 1.0f };
        auto& B = const_cast<std::vector<float>&>(fc.bias());
        B = { 1.0f, 2.0f };
    }

    // 2) Forward: x = [10,20] → y = [10*1 + 20*0 + 1, 10*0 + 20*1 + 2] = [11,22]
    std::vector<float> x = {10.0f, 20.0f};
    auto y = fc.forward(x);
    std::cout << "y = [" << y[0] << ", " << y[1] << "] (ожидаем [11,22])\n";
    assert(y[0] == 11.0f);
    assert(y[1] == 22.0f);

    // 3) Backward: пусть grad_y = [1, 1]
    std::vector<float> gy = {1.0f, 1.0f};
    auto gx = fc.backward(gy);

    //  grad_b = [1,1]
    auto gb = fc.gradBias();
    std::cout << "gradBias = [" << gb[0] << ", " << gb[1] << "] (ожидаем [1,1])\n";
    assert(gb[0] == 1.0f);
    assert(gb[1] == 1.0f);

    //  grad_w = [ x0*1, x1*1; x0*1, x1*1 ] = [10,20,10,20]
    auto gw = fc.gradWeight();
    std::cout << "gradWeight = [";
    for(int i=0;i<4;++i) std::cout << gw[i] << (i<3?", ":"");
    std::cout << "] (ожидаем [10,20,10,20])\n";
    assert(gw[0] == 10.0f);
    assert(gw[1] == 20.0f);
    assert(gw[2] == 10.0f);
    assert(gw[3] == 20.0f);

    //  grad_x = [W00*1 + W10*1, W01*1 + W11*1] = [1+0,0+1] = [1,1]
    std::cout << "gradInput = [" << gx[0] << ", " << gx[1] << "] (ожидаем [1,1])\n";
    assert(gx[0] == 1.0f);
    assert(gx[1] == 1.0f);

    // 4) zeroGrad и проверка, что обнулились
    fc.zeroGrad();
    auto gw0 = fc.gradWeight(), gb0 = fc.gradBias();
    for(float v : gw0) assert(v == 0.0f);
    for(float v : gb0) assert(v == 0.0f);

    std::cout << "[OK] FullyConnected tests passed\n";
    return 0;
}
