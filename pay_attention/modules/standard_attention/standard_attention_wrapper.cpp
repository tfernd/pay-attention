#include <torch/extension.h>
#include <vector>

template <typename T>
void standard_attention_kernel(
    const T *q,
    const T *k,
    const T *v,
    const bool *mask,
    T *out,
    int B, int T, int T_, int C, int C_);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void standard_attention_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor mask, torch::Tensor out)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(mask);
    CHECK_INPUT(out);

    int B = q.size(0);
    int T = q.size(1);
    int T_ = k.size(1);
    int C = q.size(2);
    int C_ = v.size(2);

    const int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(B, (T + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(q.type(), "standard_attention_cuda", ([&]
                                                                     { standard_attention_kernel<scalar_t><<<grid, block>>>(
                                                                           q.data_ptr<scalar_t>(),
                                                                           k.data_ptr<scalar_t>(),
                                                                           v.data_ptr<scalar_t>(),
                                                                           mask.data_ptr<bool>(),
                                                                           out.data_ptr<scalar_t>(),
                                                                           B, T, T_, C, C_); }));

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &standard_attention_cuda, "Standard Attention CUDA forward");
}
