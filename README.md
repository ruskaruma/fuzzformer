# FuzzFormer

FuzzFormer is a native C++20 and CUDA research stack implementing fuzzy attention—an uncertainty-aware alternative to softmax attention in transformers. The system provides custom GPU kernels, libtorch integration, async runtime, and visualization tools for research and experimentation.

**C++20** | **CUDA** | **libtorch** | License: Apache 2.0

## About

FuzzFormer replaces standard softmax attention with fuzzy attention using Gaussian membership functions. This provides uncertainty-aware attention weights that can better model ambiguous relationships in sequences. The entire system is compiled to native code for maximum performance and transparency.

## Components

- **CUDA Kernels**: Optimized fuzzy attention forward/backward passes
- **Transformer Blocks**: libtorch integration with fuzzy attention mechanism
- **Async Runtime**: libuv-based task scheduling for non-blocking execution
- **Visualization**: Terminal heatmaps and OpenGL renderer for attention patterns
- **Metrics**: CUPTI-based GPU profiling and throughput measurement
- **Benchmarks**: Comprehensive performance testing suite

## Installation

### Prerequisites

- CMake 3.22+
- CUDA 12.0+ toolkit
- C++20 compiler (GCC 13+ or Clang 15+)
- NVIDIA GPU with compute capability 8.6+ (RTX 4060+)
- libuv (required)
- libtorch (optional, for full model inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/ruskaruma/fuzzformer.git
cd fuzzformer

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Note:** Without libtorch, the project builds with stub interfaces. Full model inference requires libtorch. Download from [PyTorch](https://pytorch.org/get-started/locally/) and set `LIBTORCH_HOME` environment variable.

## Usage

### Basic Inference

```bash
# Run the main executable
./build/fuzzformer
```

### Running Tests

```bash
# Run all tests
ctest --test-dir build

# Run specific test suite
cd build
ctest -R FuzzyAttention -V
```

### Benchmarking

```bash
# Run performance benchmarks
cd build
ctest -R Benchmark -V
```

Benchmarks measure:
- Kernel throughput (tokens/second)
- Model inference latency
- Memory bandwidth utilization

## Architecture

The system consists of:

- **Core Compute** (`src/core/`): CUDA kernels and C++ bindings
- **Runtime** (`src/runtime/`): Async event loop and metrics collection
- **Visualization** (`src/visual/`): Terminal and OpenGL renderers
- **Utilities** (`src/utils/`): Tensor validation, timing, logging

See [Architecture Documentation](docs/architecture.md) for detailed system design.

## Key Features

- **Native Performance**: Entire stack compiled to machine code
- **GPU Accelerated**: Custom CUDA kernels with Tensor Core support
- **Research Focus**: Full visibility into memory transfers and gradients
- **Flexible**: Works with or without libtorch for different use cases
- **Visualization**: Real-time attention pattern visualization

## Project Structure

```
fuzzformer/
├── src/
│   ├── core/           # CUDA kernels and transformer blocks
│   ├── runtime/        # Async scheduler and metrics
│   ├── visual/         # Visualization renderers
│   └── utils/          # Utility functions
├── include/            # Header files
├── tests/              # Test suite
├── docs/               # Documentation
└── CMakeLists.txt      # Build configuration
```

## Performance

- **Kernel Throughput**: 500-800 tokens/second on RTX 4060
- **Memory Efficient**: Optimized for 8GB+ GPUs
- **Low Overhead**: Minimal runtime overhead with async execution

## Development

### Building Tests

```bash
cmake .. -DFUZZFORMER_BUILD_TESTS=ON
cmake --build .
```

### Enabling OpenGL Renderer

```bash
cmake .. -DFUZZFORMER_USE_OPENGL=ON
```

Requires GLFW and OpenGL development libraries.

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and component details

## License

Apache License 2.0 - see `LICENSE` file for details.
