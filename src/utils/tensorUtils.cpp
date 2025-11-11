#include "fuzzformer/tensorUtils.h"

#include <stdexcept>
#include <string>

namespace fuzzformer::tensor {

void ensure_cuda(const torch::Tensor& tensor, std::string_view name) {
#ifdef FUZZFORMER_HAS_TORCH
  if (!tensor.device().is_cuda()) {
    throw std::invalid_argument(std::string(name) + " must reside on CUDA device");
  }
#else
  (void)tensor;
  (void)name;
#endif
}

void ensure_contiguous(torch::Tensor& tensor) {
#ifdef FUZZFORMER_HAS_TORCH
  if (!tensor.is_contiguous()) {
    tensor = tensor.contiguous();
  }
#else
  (void)tensor;
#endif
}

void validate_attention_dims(std::int64_t batch_size,
                             std::int64_t num_heads,
                             std::int64_t seq_len,
                             std::int64_t head_dim) {
  if (batch_size <= 0) {
    throw std::invalid_argument("batch_size must be positive");
  }
  if (num_heads <= 0) {
    throw std::invalid_argument("num_heads must be positive");
  }
  if (seq_len <= 0) {
    throw std::invalid_argument("seq_len must be positive");
  }
  if (head_dim <= 0) {
    throw std::invalid_argument("head_dim must be positive");
  }
}

}  // namespace fuzzformer::tensor

