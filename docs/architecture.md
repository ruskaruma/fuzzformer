# FuzzFormer Architecture

## Overview

FuzzFormer is a native C++20 and CUDA research platform implementing fuzzy attention—an uncertainty-aware alternative to softmax attention in transformers. The system provides custom GPU kernels, asynchronous execution, and real-time visualization.

## System Components

### Core Compute (`src/core/`)
- **fuzzyAttention.cu**: CUDA kernels for forward/backward fuzzy attention
- **fuzzyAttention.cpp**: C++ bindings to libtorch
- **transformerBlock.cpp**: Transformer block with fuzzy attention
- **model.cpp**: Full model assembly

### Runtime (`src/runtime/`)
- **eventLoop.cpp**: libuv-based async event loop
- **asyncScheduler.cpp**: High-level task scheduler
- **metricsCollector.cpp**: Performance metrics with CUPTI integration

### Visualization (`src/visual/`)
- **terminalHeatmap.cpp**: ASCII heatmap renderer
- **renderer.cpp**: Tensor-to-heatmap conversion with OpenGL support

### Utilities (`src/utils/`)
- Tensor validation, timing, and logging utilities

## System Architecture

```mermaid
graph TB
    Main[main.cpp] --> Model[FuzzFormer Model]
    Model --> Block[Transformer Block]
    Block --> Attn[Fuzzy Attention]
    Attn --> CUDA[CUDA Kernels]
    Main --> Scheduler[AsyncScheduler]
    Scheduler --> EventLoop[EventLoop]
    Attn --> Renderer[VisualRenderer]
    Renderer --> Heatmap[Terminal/OpenGL]
```

## Data Flow

```mermaid
flowchart TD
    Input[Input Embeddings] --> QKV[Q, K, V Projections]
    QKV --> Dot[Scaled Dot Product]
    Dot --> Score[Score Matrix]
    Score --> Gaussian[Gaussian Membership]
    Gaussian --> Norm[L1 Normalization]
    Norm --> Weight[Weighted Sum with V]
    Weight --> Output[Model Output]
    
    style Input fill:#e1f5ff
    style Output fill:#d4edda
    style Gaussian fill:#fff3cd
```

## Fuzzy Attention Flow

```mermaid
graph LR
    Q[Queries] --> Dot[Dot Product]
    K[Keys] --> Dot
    Dot --> Scale[Scale]
    Scale --> Score[Score]
    Score --> Diff[Diff with Beta]
    Diff --> Exp[Gaussian Exp]
    Exp --> Mu[Membership]
    Mu --> Norm[Normalize]
    Norm --> Weight[Weight]
    Weight --> Mul[Multiply V]
    V[Values] --> Mul
    Mul --> Out[Output]
    
    style Mu fill:#fff3cd
    style Out fill:#d4edda
```

## Async Execution

```mermaid
sequenceDiagram
    participant M as Main Thread
    participant S as AsyncScheduler
    participant L as EventLoop
    participant Q as Task Queue
    participant W as Worker Thread
    
    M->>S: schedule_async
    S->>L: schedule
    L->>Q: enqueue
    L->>W: notify
    W->>Q: dequeue
    W->>W: execute
    W-->>M: complete
```

## Transformer Block

```mermaid
graph LR
    Input --> Norm1[Layer Norm]
    Norm1 --> QKV[QKV Projections]
    QKV --> FA[Fuzzy Attention]
    FA --> Add1[Add & Norm]
    Input --> Add1
    Add1 --> FFN[Feed Forward]
    FFN --> Add2[Add & Norm]
    Add1 --> Add2
    Add2 --> Output
    
    style FA fill:#fff3cd
```

## Build & Test

- **CMake 3.22+**, **CUDA 12.0+**, **libtorch** (optional), **libuv** (required)

## Performance

- Kernel fusion opportunities
- Coalesced memory access
- 128 threads per block
- Tensor Core support (future)
