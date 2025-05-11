// train.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>

#include "data/DataLoader.h"
#include "Network.h"

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Использование:\n"
                  << "  " << argv[0] << " <data_dir> <epochs> <batch_size> [val_ratio] [checkpoint_prefix]\n";
        return 1;
    }

    const std::string data_dir    = argv[1];
    const int         epochs      = std::stoi(argv[2]);
    const int         batch_size  = std::stoi(argv[3]);
    const float       val_ratio   = (argc > 4 ? std::stof(argv[4]) : 0.2f);
    const std::string ckpt_prefix = (argc > 5 ? argv[5] : std::string("ckpt"));
    const std::string ckpt_file   = ckpt_prefix + ".bin";

    // 1) Создаём загрузчик данных
    DataLoader loader(data_dir, batch_size, val_ratio);
    const size_t num_train = loader.getNumTrainSamples();
    const size_t num_val   = loader.getNumValSamples();
    const int    train_steps = static_cast<int>((num_train + batch_size - 1) / batch_size);
    const int    val_steps   = static_cast<int>((num_val   + batch_size - 1) / batch_size);

    std::cout << "Train samples: " << num_train
              << ", Val samples: " << num_val
              << ", Batch size: "   << batch_size
              << ", Steps/epoch: "  << train_steps << "\n";

    // 2) Создаём сеть
    Network net;

    // 3) Попытка загрузить предыдущий чекпоинт
    {
        std::ifstream fin(ckpt_file, std::ios::binary);
        if (fin) {
            std::cout << "Загружаем чекпоинт \"" << ckpt_file << "\"...\n";
            net.loadCheckpoint(ckpt_file);
        }
    }

    // 4) Тренировка по эпохам
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;
        int    epoch_correct = 0;
        int    total_train_samples = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        loader.reset();  // сброс курсоров train_pos_ и val_pos_

        for (int step = 0; step < train_steps; ++step) {
            // 4.1) Загружаем батч
            auto [batchX, batchY] = loader.nextBatch(true);

            // 4.2) Сбрасываем градиенты
            net.zeroGrad();

            // 4.3) Накапливаем градиенты по всем примерам батча
            for (int i = 0; i < batch_size; ++i) {
                const Tensor3D& x = batchX[i];
                int y = batchY[i];

                auto logits = net.forward(x, /*training=*/true);
                float loss = net.computeLoss({y});
                net.backward();

                epoch_loss += loss;
                total_train_samples += 1;
                if (argmax(logits) == y) {
                    epoch_correct += 1;
                }
            }

            // 4.4) Шаг оптимизации для всего батча
            net.optimize();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

        double avg_loss = epoch_loss / total_train_samples;
        double accuracy = 100.0 * epoch_correct / total_train_samples;
        std::cout << "[Epoch " << epoch << "/" << epochs << "] "
                  << "Train Loss=" << avg_loss
                  << ", Train Acc=" << accuracy << "% ("
                  << epoch_correct << "/" << total_train_samples << ") "
                  << "Time=" << train_time << "s\n";

        // 5) Валидация после каждой эпохи
        double val_loss = 0.0;
        int    val_correct = 0;
        int    total_val_samples = 0;

        // курсор val_pos_ сбросился вызовом loader.reset()
        for (int step = 0; step < val_steps; ++step) {
            auto [batchX, batchY] = loader.nextBatch(false);

            for (int i = 0; i < batch_size; ++i) {
                const Tensor3D& x = batchX[i];
                int y = batchY[i];

                auto logits = net.forward(x, /*training=*/false);
                float loss  = net.computeLoss({y});
                val_loss += loss;
                total_val_samples += 1;
                if (argmax(logits) == y) {
                    val_correct += 1;
                }
            }
        }

        double avg_val_loss = val_loss / total_val_samples;
        double val_accuracy = 100.0 * val_correct / total_val_samples;
        std::cout << "           Val   Loss=" << avg_val_loss
                  << ", Val   Acc=" << val_accuracy << "% ("
                  << val_correct << "/" << total_val_samples << ")\n";

        // 6) Сохраняем чекпоинт
        std::cout << "Сохраняем чекпоинт: " << ckpt_file << "\n";
        net.saveCheckpoint(ckpt_file);
    }

    std::cout << "Обучение завершено.\n";
    return 0;
}
