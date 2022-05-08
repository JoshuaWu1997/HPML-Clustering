#include <torch/extension.h>

void mdist_interface(float *, float *, float *, int, int, int, int);

void cdist_interface(float *, float *, float *, int, int, int, int);

torch::Tensor mdist(torch::Tensor x, torch::Tensor y, int p) {
    if (x.device().type() == torch::kCPU) {
        AT_ERROR("device not implemented");
    } else if (x.device().type() == torch::kCUDA) {
        int x_size = x.size(0);
        int y_size = y.size(0);
        int dim = x.size(1);
        auto z = torch::zeros({x_size * y_size}).to(x);
        mdist_interface(
                x.reshape({x_size * dim}).data_ptr<float>(),
                y.reshape({y_size * dim}).data_ptr<float>(),
                z.data_ptr<float>(),
                p, x_size, y_size, dim);
        return z.reshape({x_size, y_size});
    }
    AT_ERROR("No such device: ", x.device());
}

torch::Tensor cdist(torch::Tensor x, torch::Tensor y, int p) {
    if (x.device().type() == torch::kCPU) {
        AT_ERROR("device not implemented");
    } else if (x.device().type() == torch::kCUDA) {
        int x_size = x.size(0);
        int y_size = y.size(0);
        int dim = x.size(1);
        auto z = torch::zeros({x_size * y_size}).to(x);
        cdist_interface(
                x.reshape({x_size * dim}).data_ptr<float>(),
                y.reshape({y_size * dim}).data_ptr<float>(),
                z.data_ptr<float>(),
                p, x_size, y_size, dim);
        return z.reshape({x_size, y_size});
    }
    AT_ERROR("No such device: ", x.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("mdist", &mdist, "mdist");
m.def("cdist", &cdist, "cdist");
}