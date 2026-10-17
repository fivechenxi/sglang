#include <ATen/ATen.h>
#include <torch/library.h>
#include <optional>

void glm53_r37_topk(const at::Tensor&, at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);

TORCH_LIBRARY(glm53_topk_r37, m) {
  m.def("plain(Tensor scores, Tensor(a!) output, Tensor lengths, Tensor? starts) -> ()", &glm53_r37_topk);
}
