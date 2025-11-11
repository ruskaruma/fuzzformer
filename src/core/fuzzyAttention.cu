#include <cuda_runtime.h>

namespace fuzzformer {
namespace kernels {

__global__ void placeholder_kernel() {}

void launch_placeholder_kernel() {
  dim3 block(1);
  dim3 grid(1);
  placeholder_kernel<<<grid, block>>>();
}

}  // namespace kernels
}  // namespace fuzzformer

