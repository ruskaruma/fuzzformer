# FuzzFormer

FuzzFormer is a native C++20 and CUDA research stack that prototypes fuzzy attention—an uncertainty-aware alternative to softmax—for transformer workloads. The system keeps the entire execution path compiled: custom Tensor Core kernels, libtorch modules, coroutine-driven runtime, and OpenGL visualization work together to expose every memory transfer and gradient path for inspection while staying performant on modern NVIDIA GPUs.

## Components
- Core CUDA kernels for fuzzy attention forward/backward experimentation
- Libtorch transformer blocks wired to the custom attention operator
- Coroutine-based runtime skeleton for async training and rendering
- Utility layer for tensor validation, timing, and logging
- GoogleTest harness ready for CUDA-enabled regression tests

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Test
```bash
ctest --test-dir build
```

## License
See the `LICENSE` file for details.
