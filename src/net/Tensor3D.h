#pragma once

#include <vector>
#include <cassert>
#include <stdexcept>

/**
 *  Простая реализация 4D-тензора размерности
 *  depth × height × width × channels, хранящаяся
 *  в линейном буфере в порядке row-major:
 *
 *    index = ((d * H + h) * W + w) * C + c
 */
class Tensor3D {
public:
    // Конструктор: задаём shape, буфер заполняется нулями
    Tensor3D(int depth, int height, int width, int channels);

    // Пустой конструктор, shape = {0,0,0,0}
    Tensor3D() : D_(0), H_(0), W_(0), C_(0) {}

    // Доступ по ссылке с проверкой границ
    float&       operator()(int d, int h, int w, int c);
    const float& operator()(int d, int h, int w, int c) const;

    // Заполнить одним значением
    void fill(float value);

    // Получить количество измерений
    int depth()    const { return D_; }
    int height()   const { return H_; }
    int width()    const { return W_; }
    int channels() const { return C_; }

    // Общее число элементов
    int size() const { return D_*H_*W_*C_; }

    // Сырой указатель
    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Преобразовать в 1D вектор (depth*height*width*channels)
    std::vector<float> flatten() const {
        return data_;
    }

    // Переопределить размеры (reshape), не меняя порядок и объём данных
    void reshape(int newD, int newH, int newW, int newC) {
        if (newD*newH*newW*newC != size())
            throw std::runtime_error("Tensor3D::reshape: total size mismatch");
        D_=newD; H_=newH; W_=newW; C_=newC;
    }

private:
    int D_, H_, W_, C_;
    std::vector<float> data_;
};
