#include "Tensor3D.h"
#include <algorithm>

Tensor3D::Tensor3D(int depth, int height, int width, int channels)
    : D_(depth), H_(height), W_(width), C_(channels),
      data_(static_cast<size_t>(depth)*height*width*channels, 0.0f)
{
    assert(depth > 0 && height > 0 && width > 0 && channels > 0);
}

inline int calc_index(int d, int h, int w, int c, int H, int W, int C) {
    return ((d * H + h) * W + w) * C + c;
}

float& Tensor3D::operator()(int d, int h, int w, int c) {
    assert(d >= 0 && d < D_);
    assert(h >= 0 && h < H_);
    assert(w >= 0 && w < W_);
    assert(c >= 0 && c < C_);
    return data_[calc_index(d, h, w, c, H_, W_, C_)];
}

const float& Tensor3D::operator()(int d, int h, int w, int c) const {
    assert(d >= 0 && d < D_);
    assert(h >= 0 && h < H_);
    assert(w >= 0 && w < W_);
    assert(c >= 0 && c < C_);
    return data_[calc_index(d, h, w, c, H_, W_, C_)];
}

void Tensor3D::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}
