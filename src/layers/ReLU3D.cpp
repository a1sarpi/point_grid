#include "layers/ReLU3D.h"
#include <cassert>

Tensor3D ReLU3D::forward(const Tensor3D& x) {
    D_ = x.depth(); H_ = x.height();
    W_ = x.width(); C_ = x.channels();

    int N = D_*H_*W_*C_;
    Tensor3D y(D_, H_, W_, C_);
    mask_.assign(N, 0);

    auto x_data = x.data();
    auto y_data = y.data();
    for(int i = 0; i < N; ++i) {
        if (x_data[i] > 0.0f) {
            y_data[i]    = x_data[i];
            mask_[i]     = 1;
        } else {
            y_data[i]    = 0.0f;
            // mask_[i] already 0
        }
    }
    return y;
}

Tensor3D ReLU3D::backward(const Tensor3D& grad_y) {
    assert(grad_y.depth()==D_ && grad_y.height()==H_
        && grad_y.width()==W_ && grad_y.channels()==C_);

    Tensor3D grad_x(D_, H_, W_, C_);
    auto gy = grad_y.data();
    auto gx = grad_x.data();
    int N = D_*H_*W_*C_;
    for(int i = 0; i < N; ++i) {
        gx[i] = mask_[i] ? gy[i] : 0.0f;
    }
    return grad_x;
}

void ReLU3D::zeroGrad() {
    mask_.clear();
    D_ = H_ = W_ = C_ = 0;
}
