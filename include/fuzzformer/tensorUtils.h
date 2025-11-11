#pragma once

#include <cstdint>
#include <string_view>

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#else
#include "fuzzformer/torchStub.h"
#endif

namespace fuzzformer::tensor {

void ensure_cuda(const torch::Tensor& tensor, std::string_view name);

void ensure_contiguous(torch::Tensor& tensor);

void validate_attention_dims(std::int64_t batch_size,
                             std::int64_t num_heads,
                             std::int64_t seq_len,
                             std::int64_t head_dim);

}  // namespace fuzzformer::tensor

