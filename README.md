# FuzzFormer

FuzzFormer is a native C++20 and CUDA research stack implementing fuzzy attention—an uncertainty-aware alternative to softmax attention in transformers. The system provides custom GPU kernels, libtorch integration, async runtime, and visualization tools for research and experimentation.

## Components

- **CUDA Kernels**: Fuzzy attention forward/backward passes
- **Transformer Blocks**: libtorch integration with fuzzy attention
- **Async Runtime**: libuv-based task scheduling
- **Visualization**: Terminal heatmaps and OpenGL renderer
- **Metrics**: CUPTI-based GPU profiling and throughput measurement
- **Benchmarks**: Comprehensive performance testing suite

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

**Note:** Some tests are skipped when libtorch is not available.

## Benchmark

```bash
cd build
ctest -R Benchmark -V
```

Benchmarks measure kernel throughput, model inference latency, and memory bandwidth.

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and component details

## License

See the `LICENSE` file for details.
