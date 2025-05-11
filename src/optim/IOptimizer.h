#ifndef IOPTIMIZER_H
#define IOPTIMIZER_H

#include <vector>

/**
 * \brief Абстрактный интерфейс оптимизатора.
 *
 * Реализует регистрацию параметров с градиентами,
 * выполнение шага оптимизации и обнуление градиентов.
 */
class IOptimizer {
public:
    /**
     * \brief Зарегистрировать параметры и соответствующие им градиенты.
     * \param param Ссылка на вектор параметров.
     * \param grad Ссылка на вектор градиентов тех же размеров.
     */
    virtual void addParam(std::vector<float>& param,
                          std::vector<float>& grad) = 0;

    /**
     * \brief Сделать один шаг обновления всех зарегистрированных параметров.
     */
    virtual void step() = 0;

    /**
     * \brief Обнулить все градиенты для подготовки к следующей итерации.
     */
    virtual void zeroGrad() = 0;

    virtual ~IOptimizer() = default;
};

#endif // IOPTIMIZER_H
