#include "BatchNorm3D.h"
#include <cmath>

BatchNorm3D::BatchNorm3D(int channels, float eps, float momentum)
    : C_(channels), eps_(eps), momentum_(momentum),
      gamma_(channels, 1.0f), beta_(channels, 0.0f),
      grad_gamma_(channels, 0.0f), grad_beta_(channels, 0.0f),
      running_mean_(channels, 0.0f), running_var_(channels, 1.0f),
      D_(0), H_(0), W_(0), N_(0)
{}

// forward
Tensor3D BatchNorm3D::forward(const Tensor3D& x, bool training) {
    // Проверяем число каналов
    assert(x.channels() == C_);
    D_ = x.depth(); H_ = x.height(); W_ = x.width();
    N_ = D_ * H_ * W_;

    Tensor3D y(D_, H_, W_, C_);

    // Буферы
    batch_mean_.assign(C_, 0.0f);
    batch_var_.assign(C_,  0.0f);
    inv_std_.assign(C_,    0.0f);
    x_hat_.assign(N_ * C_, 0.0f);

    const float* xdata = x.data();
    float* ydata = y.data();

    // 1) Вычисляем batch_mean
    for(int c=0; c<C_; ++c) {
        float sum = 0;
        for(int i=0; i<N_; ++i) {
            sum += xdata[i*C_ + c];
        }
        batch_mean_[c] = sum / N_;
    }

    // 2) Вычисляем batch_var
    for(int c=0; c<C_; ++c) {
        float sum2 = 0;
        float mu = batch_mean_[c];
        for(int i=0; i<N_; ++i) {
            float v = xdata[i*C_ + c] - mu;
            sum2 += v * v;
        }
        batch_var_[c] = sum2 / N_;
        inv_std_[c] = 1.0f / std::sqrt(batch_var_[c] + eps_);
    }

    // 3) Обновляем running-статистики, если training
    if(training) {
        for(int c=0; c<C_; ++c) {
            running_mean_[c] = momentum_*batch_mean_[c]
                             + (1-momentum_)*running_mean_[c];
            running_var_[c]  = momentum_*batch_var_[c]
                             + (1-momentum_)*running_var_[c];
        }
    }

    // 4) Нормализация и scale+shift
    for(int i=0; i<N_; ++i) {
        for(int c=0; c<C_; ++c) {
            int idx = i*C_ + c;
            float mu = batch_mean_[c];
            float is = inv_std_[c];
            float xh = (xdata[idx] - mu) * is;
            x_hat_[idx] = xh;
            ydata[idx]   = gamma_[c]*xh + beta_[c];
        }
    }

    return y;
}

// backward
Tensor3D BatchNorm3D::backward(const Tensor3D& grad_y) {
    // Проверяем, что forward был вызван
    assert(D_>0 && N_>0);
    assert(grad_y.depth()==D_ && grad_y.height()==H_
        && grad_y.width()==W_ && grad_y.channels()==C_);

    Tensor3D grad_x(D_, H_, W_, C_);
    const float* dy = grad_y.data();
    float* dx = grad_x.data();

    // Сбрасываем градиенты по γ/β
    std::fill(grad_gamma_.begin(), grad_gamma_.end(), 0.0f);
    std::fill(grad_beta_.begin(),  grad_beta_.end(),  0.0f);

    // 1) grad_beta и grad_gamma
    for(int c=0; c<C_; ++c){
        float gb=0, gg=0;
        for(int i=0; i<N_; ++i){
            int idx = i*C_ + c;
            gb += dy[idx];
            gg += dy[idx] * x_hat_[idx];
        }
        grad_beta_[c]  = gb;
        grad_gamma_[c] = gg;
    }

    // 2) grad_x по формуле
    for(int c=0; c<C_; ++c){
        float g  = grad_gamma_[c];
        float b  = grad_beta_[c];
        float is = inv_std_[c];
        float scale = gamma_[c] * is / N_;
        for(int i=0; i<N_; ++i){
            int idx = i*C_ + c;
            // N*dy - grad_beta - x_hat*grad_gamma
            dx[idx] = scale * ( N_*dy[idx]
                              - b
                              - x_hat_[idx] * g );
        }
    }

    return grad_x;
}

// сброс внутренних состояний (кроме running_*)
void BatchNorm3D::zeroGrad() {
    std::fill(grad_gamma_.begin(), grad_gamma_.end(), 0.0f);
    std::fill(grad_beta_.begin(),  grad_beta_.end(),  0.0f);
    batch_mean_.clear();
    batch_var_.clear();
    inv_std_.clear();
    x_hat_.clear();
    D_ = H_ = W_ = N_ = 0;
}
