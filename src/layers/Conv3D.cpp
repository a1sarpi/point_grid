#include "Conv3D.h"
#include <cmath>
#include <random>

Conv3D::Conv3D(int in_ch, int out_ch,
               int kD, int kH, int kW,
               int sD, int sH, int sW,
               Padding pad)
    : in_ch_(in_ch), out_ch_(out_ch),
      kD_(kD), kH_(kH), kW_(kW),
      sD_(sD), sH_(sH), sW_(sW),
      pad_(pad)
{
    int kernel_size = kD_*kH_*kW_*in_ch_*out_ch_;
    weight_.assign(kernel_size, 0.0f);
    bias_.assign(out_ch_, 0.0f);
    grad_w_.assign(kernel_size, 0.0f);
    grad_b_.assign(out_ch_, 0.0f);
    initWeightsXavier();
}

void Conv3D::initWeightsXavier() {
    float fan_in  = in_ch_ * kD_ * kH_ * kW_;
    float fan_out = out_ch_ * kD_ * kH_ * kW_;
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::mt19937 gen(std::random_device{}());
    for (auto& w : weight_) w = dist(gen);
    for (auto& b : bias_)   b = 0.0f;
}

void Conv3D::computeOutputDims(int D, int H, int W,
                               int& D_out, int& H_out, int& W_out) const
{
    auto dim = [&](int size, int k, int s){
        if (pad_ == Padding::SAME)
            return (size + s - 1) / s;
        else // VALID
            return (size - k + s) / s;
    };
    D_out = dim(D, kD_, sD_);
    H_out = dim(H, kH_, sH_);
    W_out = dim(W, kW_, sW_);
}

Tensor3D Conv3D::forward(const Tensor3D& x) {
    int D = x.depth(), H = x.height(), W = x.width();
    int D_out, H_out, W_out;
    computeOutputDims(D, H, W, D_out, H_out, W_out);
    Tensor3D y(D_out, H_out, W_out, out_ch_);

    // for each output voxel
    for(int od=0; od<D_out; ++od){
        for(int oh=0; oh<H_out; ++oh){
            for(int ow=0; ow<W_out; ++ow){
                for(int oc=0; oc<out_ch_; ++oc){
                    float sum = bias_[oc];
                    for(int kd=0; kd<kD_; ++kd){
                        int id = od*sD_ + kd - (pad_==Padding::SAME ? kD_/2 : 0);
                        for(int kh=0; kh<kH_; ++kh){
                            int ih = oh*sH_ + kh - (pad_==Padding::SAME ? kH_/2 : 0);
                            for(int kw=0; kw<kW_; ++kw){
                                int iw = ow*sW_ + kw - (pad_==Padding::SAME ? kW_/2 : 0);
                                if(id<0||id>=D||ih<0||ih>=H||iw<0||iw>=W) continue;
                                for(int ic=0; ic<in_ch_; ++ic){
                                    float xv = x(id,ih,iw,ic);
                                    float wv = weight_[wIndex(kd,kh,kw,ic,oc)];
                                    sum += xv * wv;
                                }
                            }
                        }
                    }
                    y(od,oh,ow,oc) = sum;
                }
            }
        }
    }
    return y;
}

Tensor3D Conv3D::backward(const Tensor3D& x, const Tensor3D& grad_out) {
    // Аналогично forward: считаем grad_w_, grad_b_ и grad_input
    int D = x.depth(), H = x.height(), W = x.width();
    int D_out, H_out, W_out;
    computeOutputDims(D, H, W, D_out, H_out, W_out);

    Tensor3D grad_in(D, H, W, in_ch_);
    grad_in.fill(0.0f);

    // Сбросим прежние градиенты параметров
    // (!! обычно zeroGrad вызывают отдельно)
    //zeroGrad();

    // Вычисляем bias градиенты
    for(int od=0; od<D_out; ++od)
    for(int oh=0; oh<H_out; ++oh)
    for(int ow=0; ow<W_out; ++ow)
    for(int oc=0; oc<out_ch_; ++oc){
        grad_b_[oc] += grad_out(od,oh,ow,oc);
    }

    // Вычисляем весовые градиенты и grad_input
    for(int od=0; od<D_out; ++od){
        for(int oh=0; oh<H_out; ++oh){
            for(int ow=0; ow<W_out; ++ow){
                for(int oc=0; oc<out_ch_; ++oc){
                    float go = grad_out(od,oh,ow,oc);
                    for(int kd=0; kd<kD_; ++kd){
                        int id = od*sD_ + kd - (pad_==Padding::SAME? kD_/2:0);
                        for(int kh=0; kh<kH_; ++kh){
                            int ih = oh*sH_ + kh - (pad_==Padding::SAME? kH_/2:0);
                            for(int kw=0; kw<kW_; ++kw){
                                int iw = ow*sW_ + kw - (pad_==Padding::SAME? kW_/2:0);
                                if(id<0||id>=D||ih<0||ih>=H||iw<0||iw>=W) continue;
                                for(int ic=0; ic<in_ch_; ++ic){
                                    int widx = wIndex(kd,kh,kw,ic,oc);
                                    float xv = x(id,ih,iw,ic);
                                    grad_w_[widx] += xv * go;
                                    grad_in(id,ih,iw,ic) += weight_[widx] * go;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_in;
}

void Conv3D::zeroGrad() {
    std::fill(grad_w_.begin(), grad_w_.end(), 0.0f);
    std::fill(grad_b_.begin(), grad_b_.end(), 0.0f);
}
