#include <ATen/ATen.h>
#include <torch/library.h>
#include <optional>

// Distinct symbols prevent accidental binding to the installed baseline .so.
void glm53_r32_topk(const at::Tensor&, at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);
void glm53_r32_paged(const at::Tensor&, const at::Tensor&, at::Tensor&,
                    const at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);
void glm53_r32_ragged(const at::Tensor&, const at::Tensor&, at::Tensor&,
                     const at::Tensor&, std::optional<at::Tensor>);

TORCH_LIBRARY(glm53_topk_r32, m) {
  m.def("plain(Tensor scores, Tensor(a!) output, Tensor lengths, Tensor? starts) -> ()", &glm53_r32_topk);
  m.def("paged(Tensor scores, Tensor lengths, Tensor(a!) output, Tensor pages, Tensor cu, Tensor? starts) -> ()", &glm53_r32_paged);
  m.def("ragged(Tensor scores, Tensor lengths, Tensor(a!) output, Tensor offsets, Tensor? starts) -> ()", &glm53_r32_ragged);
}
