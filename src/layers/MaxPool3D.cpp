#include "layers/MaxPool3D.h"
#include <limits>       // для numeric_limits
#include <cmath>
#include <algorithm>

// Целочисленный ceil‐делитель
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

MaxPool3D::MaxPool3D(int kD, int kH, int kW,
                     int sD, int sH, int sW,
                     Padding pad)
    : kD_(kD), kH_(kH), kW_(kW),
      sD_(sD), sH_(sH), sW_(sW),
      pad_(pad)
{}

Tensor3D MaxPool3D::forward(const Tensor3D& x) {
    // Сохраняем форму входа
    inD_ = x.depth(); inH_ = x.height(); inW_ = x.width(); inC_ = x.channels();

    // Вычисляем форму выхода
    if (pad_ == Padding::SAME) {
        outD_ = ceil_div(inD_, sD_);
        outH_ = ceil_div(inH_, sH_);
        outW_ = ceil_div(inW_, sW_);
    } else {
        outD_ = (inD_ - kD_) / sD_ + 1;
        outH_ = (inH_ - kH_) / sH_ + 1;
        outW_ = (inW_ - kW_) / sW_ + 1;
    }

    // Резервируем результат и индексы
    Tensor3D y(outD_, outH_, outW_, inC_);
    maxIndex_.assign(outD_*outH_*outW_*inC_, 0);

    // Сбрасываем предыдущие градиенты
    gradInput_ = Tensor3D(inD_, inH_, inW_, inC_);
    gradInput_.fill(0.0f);

    // Основной цикл по выходным элементам
    for(int d=0; d<outD_; ++d){
      for(int h=0; h<outH_; ++h){
        for(int w=0; w<outW_; ++w){
          for(int c=0; c<inC_; ++c){
            float best = -std::numeric_limits<float>::infinity();
            int bestIdx = 0;

            // Перебираем окно
            for(int kd=0; kd<kD_; ++kd){
              int id = d*sD_ + kd - (pad_==Padding::SAME ? kD_/2 : 0);
              for(int kh=0; kh<kH_; ++kh){
                int ih = h*sH_ + kh - (pad_==Padding::SAME ? kH_/2 : 0);
                for(int kw=0; kw<kW_; ++kw){
                  int iw = w*sW_ + kw - (pad_==Padding::SAME ? kW_/2 : 0);
                  // Проверяем границы
                  if(id<0||id>=inD_||ih<0||ih>=inH_||iw<0||iw>=inW_) continue;
                  int idx = ((id*inH_ + ih)*inW_ + iw)*inC_ + c;
                  float v = x.data()[idx];
                  if(v > best){
                    best = v;
                    bestIdx = idx;
                  }
                }
              }
            }

            int outFlat = ((d*outH_ + h)*outW_ + w)*inC_ + c;
            y.data()[outFlat]       = best;
            maxIndex_[outFlat]      = bestIdx;
          }
        }
      }
    }

    return y;
}

Tensor3D MaxPool3D::backward(const Tensor3D& grad_y) {
    // Сбрасываем накопленный gradInput_
    gradInput_.fill(0.0f);

    // Разбрасываем градиенты туда, где был максимум
    int N = outD_*outH_*outW_*inC_;
    for(int i=0; i<N; ++i){
        float g = grad_y.data()[i];
        gradInput_.data()[ maxIndex_[i] ] += g;
    }
    return gradInput_;
}

void MaxPool3D::zeroGrad() {
    gradInput_.fill(0.0f);
    // maxIndex_ будет перезаписан в следующем forward()
}
