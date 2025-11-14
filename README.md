# FuzzFormer

     FuzzFormer is a native C++20 and CUDA research stack that prototypes fuzzy attention—an uncertainty-aware alternative to softmax—for transformer workloads. The system keeps the entire execution path compiled: custom Tensor Core kernels, libtorch modules, coroutine-driven runtime, and OpenGL visualization work together to expose every memory transfer and gradient path for inspection while staying performant on modern NVIDIA GPUs.

## Components
- Core CUDA kernels for fuzzy attention forward/backward experimentation
- Libtorch transformer blocks wired to the custom attention operator
- libuv-based async runtime for non-blocking task execution
- Terminal heatmap visualization for attention matrices
- Performance metrics collector with throughput cards
- Comprehensive benchmarking suite
- Utility layer for tensor validation, timing, and logging
- GoogleTest harness with 17+ tests

## Dependencies

### Required
- CMake 3.22+
- CUDA 12.0+ toolkit
- C++20 compiler (GCC 13+ or Clang 15+)
- NVIDIA GPU with compute capability 8.6+ (RTX 4060+)

### Optional (for full functionality)
- **libtorch** (C++11 ABI, CUDA-enabled): Download from [PyTorch](https://pytorch.org/get-started/locally/)
  - Extract and set: `export LIBTORCH_HOME=/path/to/libtorch`
  - Configure with: `cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_HOME} ..`

**Note:** Without libtorch, the project builds with stub interfaces for testing the build system. Full model inference requires libtorch.

## Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Test
```bash
ctest --test-dir build
```

**Note:** Some tests are skipped when libtorch is not available. This is expected behavior.

## Benchmark
```bash
cd build
ctest -R Benchmark -V
```

Benchmarks measure kernel throughput, model inference latency, and memory bandwidth.

## Documentation
- [Architecture Overview](docs/architecture.md) - System design and component details

### Known Warnings
- **nvlink warnings** about incompatible static libraries (`librt.a`, `libpthread.a`, `libdl.a`) are harmless and can be ignored. They occur because CUDA's linker skips system static libs that aren't needed for device code linking.

## License
See the `LICENSE` file for details.
