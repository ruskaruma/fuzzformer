# FuzzFormer Architecture

## Overview

FuzzFormer is a native C++20 and CUDA research platform implementing fuzzy attention—an uncertainty-aware alternative to softmax attention in transformers. The system is designed as a full-stack research tool with custom GPU kernels, asynchronous execution, and real-time visualization.

## System Components

### 1. Core Compute Layer

**Location:** `src/core/`

- **fuzzyAttention.cu**: CUDA kernels implementing fuzzy attention forward and backward passes
  - Forward: Computes Gaussian membership functions, L1 normalization, weighted value aggregation
  - Backward: Gradient computation for Q, K, V, alpha, and beta parameters
  
- **fuzzyAttention.cpp**: C++ bindings connecting CUDA kernels to libtorch tensors
  - Input validation and tensor shape checking
  - CUDA stream management
  - Error handling and reporting

- **transformerBlock.cpp**: Transformer encoder block using fuzzy attention
  - Multi-head attention with fuzzy attention mechanism
  - Residual connections and layer normalization
  - Feed-forward network integration

- **model.cpp**: Full model assembly
  - Stacked transformer blocks
  - Input/output projections
  - Configurable architecture

### 2. Runtime Layer

**Location:** `src/runtime/`

- **eventLoop.cpp**: libuv-based asynchronous event loop
  - Task queue management
  - Non-blocking task execution
  - Thread-safe scheduling

- **asyncScheduler.cpp**: High-level async task scheduler
  - Wraps event loop in worker thread
  - Provides `schedule_async()` interface
  - Automatic cleanup on destruction

- **metricsCollector.cpp**: Performance metrics collection
  - Kernel timing measurement
  - Throughput card rendering
  - Extensible for CUPTI integration

### 3. Visualization Layer

**Location:** `src/visual/`

- **terminalHeatmap.cpp**: ASCII terminal heatmap renderer
  - 9-level intensity mapping (`. :-=+*#%@`)
  - 2D matrix and 1D vector visualization
  - Real-time attention weight display

- **renderer.cpp**: Tensor-to-heatmap conversion
  - libtorch tensor extraction
  - CPU transfer and formatting
  - Integration with heatmap renderer

### 4. Utilities

**Location:** `src/utils/`

- **tensorUtils.cpp**: Tensor validation and dimension checking
- **timer.cpp**: High-resolution timing utilities
- **logger.cpp**: Thread-safe logging with severity levels

## System Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        Main[main.cpp]
    end
    
    subgraph "Model Layer"
        Model[FuzzFormer Model]
        Block[Transformer Block]
        Attn[Fuzzy Attention]
    end
    
    subgraph "Runtime Layer"
        Scheduler[AsyncScheduler]
        EventLoop[EventLoop libuv]
        Metrics[MetricsCollector]
    end
    
    subgraph "Visualization Layer"
        Renderer[VisualRenderer]
        Heatmap[Terminal Heatmap]
        OpenGL[OpenGL Renderer]
    end
    
    subgraph "Core Compute"
        CUDA[CUDA Kernels]
        Bindings[C++ Bindings]
    end
    
    subgraph "Utilities"
        Utils[Tensor Utils]
        Timer[Timer]
        Logger[Logger]
    end
    
    Main --> Model
    Model --> Block
    Block --> Attn
    Attn --> CUDA
    CUDA --> Bindings
    Main --> Scheduler
    Scheduler --> EventLoop
    Main --> Metrics
    Attn --> Renderer
    Renderer --> Heatmap
    Renderer --> OpenGL
    Model --> Utils
    Metrics --> Timer
    Main --> Logger
```

## Data Flow

```mermaid
flowchart TD
    Input[Input Embeddings] --> Stack[Transformer Block Stack]
    Stack --> QKV[Q, K, V Projections]
    QKV --> Dot[Scaled Dot Product QK^T]
    Dot --> Score[Score Matrix S]
    Score --> Gaussian[Gaussian Membership μ = e^(-α(S-β)²)]
    Gaussian --> Norm[L1 Normalization]
    Norm --> Weight[Weighted Sum with V]
    Weight --> Output[Output O]
    Output --> Proj[Output Projections]
    Proj --> Result[Model Output]
    
    style Input fill:#e1f5ff
    style Result fill:#d4edda
    style Gaussian fill:#fff3cd
```

## Fuzzy Attention Flow

```mermaid
graph LR
    Q[Queries Q] --> Dot[Dot Product QK^T]
    K[Keys K] --> Dot
    Dot --> Scale[Scale by 1/√d]
    Scale --> Score[Score S]
    Score --> Diff[Diff = S - β]
    Diff --> Square[Square diff²]
    Square --> Exp[Exp -α·diff²]
    Exp --> Mu[Membership μ]
    Mu --> Sum[Sum Row]
    Sum --> Inv[1/Sum]
    Inv --> Weight[Weight = μ·inv]
    Weight --> Mul[Multiply with V]
    V[Values V] --> Mul
    Mul --> Out[Output O]
    
    style Mu fill:#fff3cd
    style Out fill:#d4edda
```

## Memory Hierarchy

```mermaid
graph TB
    DRAM[Global Memory DRAM] --> L2[L2 Cache]
    L2 --> Shared[Shared Memory per-SM]
    Shared --> Reg[Registers per-thread]
    Reg --> TC[Tensor Cores Compute]
    
    style DRAM fill:#ffcccc
    style L2 fill:#ffffcc
    style Shared fill:#ccffcc
    style Reg fill:#ccccff
    style TC fill:#ffccff
```

## Async Execution Model

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant Sched as AsyncScheduler
    participant Loop as EventLoop libuv
    participant Queue as Task Queue
    participant Worker as Worker Thread
    
    Main->>Sched: schedule_async(task)
    Sched->>Loop: schedule(task)
    Loop->>Queue: enqueue(task)
    Loop->>Worker: uv_async_send()
    Worker->>Queue: dequeue(task)
    Worker->>Worker: execute(task)
    Worker-->>Main: task complete
```

## Transformer Block Layout

```mermaid
graph LR
    Input[Input] --> Norm1[Layer Norm]
    Norm1 --> QKV[QKV Projections]
    QKV --> FA[Fuzzy Attention]
    FA --> Add1[Add & Norm]
    Input --> Add1
    Add1 --> FFN[Feed Forward]
    FFN --> Add2[Add & Norm]
    Add1 --> Add2
    Add2 --> Output[Output]
    
    style FA fill:#fff3cd
    style Output fill:#d4edda
```

## Compute Pipeline

```mermaid
flowchart LR
    subgraph "CUDA Device"
        Load[Load Q, K, V]
        Compute[Compute QK^T]
        Apply[Apply Membership]
        Normalize[Normalize Rows L1]
        Matmul[Matmul MV]
    end
    
    Load --> Compute
    Compute --> Apply
    Apply --> Normalize
    Normalize --> Matmul
    Matmul --> Result[Result]
    
    style Apply fill:#fff3cd
```

## Visualization Loop

```mermaid
flowchart LR
    Train[Training Loop] --> Update[Update Metrics]
    Update --> Generate[Generate Attention Matrix]
    Generate --> Render[Render in Terminal/OpenGL]
    Render --> Display[Display Heatmap]
    Display --> Train
    
    style Render fill:#fff3cd
```

## Kernel Fusion Flow

```mermaid
graph TD
    Dot[Dot Product Kernel] --> Mem[Membership Kernel]
    Mem --> Norm[Normalization Kernel]
    Norm --> Fused[Fused FuzzyAttentionKernel]
    
    style Fused fill:#d4edda
```

## Build System

- **CMake 3.22+**: Build configuration
- **CUDA 12.0+**: GPU kernel compilation
- **libtorch**: Optional model execution (stubs when unavailable)
- **libuv**: Async event loop (required)
- **GoogleTest**: Test framework

## Testing Strategy

- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end model inference
- **Performance Tests**: Kernel timing and throughput
- **Skip Logic**: Tests requiring libtorch skip gracefully when unavailable

## Extension Points

1. **CUPTI Integration**: Add to `metricsCollector.cpp` for detailed GPU profiling
2. **OpenGL Renderer**: Extend `visualRenderer.h` for real-time 3D visualization
3. **Additional Kernels**: Add new CUDA files following `fuzzyAttention.cu` pattern
4. **Custom Optimizers**: Extend model training loop in `main.cpp`

## Performance Considerations

- **Kernel Fusion**: Forward and backward passes are separate kernels (can be fused)
- **Memory Access**: Coalesced global memory access patterns
- **Occupancy**: 128 threads per block, configurable
- **Tensor Cores**: Architecture supports Tensor Core operations (future optimization)

