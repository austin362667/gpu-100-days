# Day 2: The LLM System - A High-Level Blueprint

Welcome to Day 2 of the GPU Challenge! Before we dive into CUDA kernels and memory optimization, we need a clear view of the system we're building. 

Large Language Models (LLMs) aren't just a single neural network—they're complex systems designed to run massive models efficiently. The goal: maximize **throughput** (tokens per second) and minimize **latency** (response time). But here's the thing: to understand where GPUs fit, we need to think like system architects, not just programmers.

Think of an LLM system as a three-story building. Each floor has a different purpose, different tools, and different optimization strategies. Our GPU programming journey will take us from the basement (raw hardware) to the penthouse (distributed systems). Let's tour the building.

### The Three Layers of an LLM System

An LLM system can be broken into three layers, each addressing a distinct challenge in running large models at scale.

| Layer | Operations | Key Focus | Representative Tools |
|-------|------------|-----------|---------------------|
| **Kernel** | Scalar/Vector/Tile instructions | Micro-architecture optimization | CUDA C/C++, Triton, PTX, CUTLASS, Mojo |
| **Graph** | Tensor primitives | Model graph optimization | PyTorch, TensorRT, ONNX, JAX, TinyGrad |
| **System** | Sharding, Batching, Orchestration | Multi-GPU coordination | SGLang, vLLM, TensorRT-LLM, DeepSpeed, Megatron-LM |

**1. The Kernel Layer: Raw GPU Execution**

This is the basement—where code meets silicon. A kernel is the smallest unit of work that runs directly on the GPU's cores. Think matrix multiplication, attention computation, or element-wise operations. The focus is on maximizing performance at the micro-architecture level.

*What's Happening?* Kernels are optimized to achieve high arithmetic intensity (computations per byte of memory accessed). Since memory access is slower than computation, the goal is to perform as much math as possible per data fetch—just like we learned with our GPU monster yesterday.

#### Kernel Layer Optimization Techniques

The kernel layer uses three main optimization strategies:

| Technique | Description | Examples |
|-----------|-------------|----------|
| **Data Locality** | Keep data close to compute units | Use registers, shared memory (including tiling), L1/L2 cache |
| **Data Movement Efficiency** | Optimize memory access patterns | Swizzling, coalescing, overlapping compute with memory |
| **Special Instructions** | Leverage hardware accelerators | TensorCore MMA, Hopper TMA, vectorized operations |

**2. The Graph Layer: Model Optimization**

This is the main floor—where we view the model as a computational graph. Instead of optimizing individual kernels, the focus is on streamlining the entire sequence of operations. Think of it as optimizing the recipe, not just the individual cooking techniques.

*What's Happening?* Frameworks analyze the graph and rewrite it for efficiency, reducing redundant computations and memory usage. They ask questions like: "Can we combine these operations? Can we eliminate this memory copy? Can we compute this once instead of three times?"

#### Graph Layer Optimization Techniques

The graph layer employs several key optimization strategies:

| Technique | Description | Impact |
|-----------|-------------|---------|
| **Operator Fusion** | Combine multiple operations into single kernels | Reduces memory I/O, eliminates intermediate writes |
| **Merging** | Mathematically combine operations (e.g., Conv+BatchNorm) | Reduces computation and memory usage |
| **Quantization** | Use lower precision (FP16, INT8, FP4) | Increases throughput, reduces memory |
| **Sparsity** | Skip zero computations (static/dynamic) | Reduces computation for sparse models |
| **JIT Compilation** | Convert dynamic graphs to optimized static graphs | Eliminates Python overhead, enables optimizations |

---

> **Personal Thoughts:** In theory, a “smart-enough” compiler could use hardware specs and algorithmic needs to auto‑generate high‑performance kernels, letting users work purely at the graph level. But no such compiler exists today, so researchers and engineers must continue hand‑tuning kernels—for the foreseeable future. I highly recommend **[Chris Lattner](https://www.nondot.org/sabre/)**’s blog series **[Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai)** at [Modular](https://www.modular.com/mojo). Besides, cutting training and inference costs remains vital: GPU FLOPS keep outpacing HBM bandwidth growth, so manual effort to turn memory‑bound tasks into compute‑bound ones is still essential. Just a late-night thought I had—happy to hear any feedback or continue the discussion! Feel free to reach out.

---

**3. The System Layer: Multi-GPU Orchestration**

This is the penthouse—where we coordinate entire clusters of GPUs. Massive models like GPT-4 or Claude exceed the capacity of a single GPU. This layer treats the model as a distributed program running across hundreds or thousands of GPUs.

*What's Happening?* The system manages compute, memory, and communication at scale. It's like conducting an orchestra where each musician (GPU) must play their part perfectly, and the conductor must ensure they're all synchronized.

*Key Techniques:*
- **Parallelism Strategies**
- **Intelligent Batching:** Like vLLM uses _PagedAttention_ to reduce KV cache fragmentation and SGLang uses _RadixAttention_ to increase  KV cache reuse. Both are sophisticated ways of promote efficient use of KV cache.

- **Communication Optimization:** Using high-speed interconnects (NVLink, InfiniBand) to minimize data transfer overhead within nodes and networks. Other than that, frameworks like [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) is a good example of overlapping communication stages of aggregate operations.

#### System Layer Parallelism Strategies

The system layer coordinates multiple GPUs using various parallelism techniques:

| Parallelism Type | How It Works | Latency Impact | Memory Impact | Communication Cost |
|------------------|--------------|----------------|---------------|--------------------|
| **Data Parallel** | Same model, different batches | ❌ No improvement | ❌ Full model per GPU | ✅ Low (inference) |
| **Pipeline Parallel** | Different layers on different GPUs | ❌ No improvement | ✅ Saves memory | ✅ Low |
| **Tensor Parallel** | Split weight matrices across GPUs | ✅ Improves latency | ✅ Saves memory | ❌ High |
| **Expert Parallel** | Distribute MoE experts across GPUs | ✅ Improves latency (large batch) | ✅ Saves memory | 🔶 Medium |
| **Sequence Parallel** | Split along sequence dimension | ✅ Improves latency (long context) | ✅ Saves memory | ❌ High |

### System Performance Metrics

Different layers optimize for different metrics:

| Layer | Primary Metrics | Secondary Metrics |
|-------|----------------|-------------------|
| **Kernel** | FLOPS, Memory Bandwidth Utilization | Latency per operation |
| **Graph** | Model FLOPS Utilization (MFU) | Memory efficiency, Compilation time |
| **System** | Tokens Per Second (TPS), Time To First Token (TTFT) | Time Per Output Token (TPOT) |

### LLM System Architecture Overview

Here's how the three layers interact in a complete LLM system:

```
┌─────────────────────────────────────────────────────────────┐  
│                    SYSTEM LAYER                             │  
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  
│  │   GPU 1     │  │   GPU 2     │  │   GPU N     │        │  
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │        │  
│  │ │ GRAPH   │ │  │ │ GRAPH   │ │  │ │ GRAPH   │ │        │  
│  │ │ LAYER   │ │  │ │ LAYER   │ │  │ │ LAYER   │ │        │  
│  │ │┌───────┐│ │  │ │┌───────┐│ │  │ │┌───────┐│ │        │  
│  │ ││KERNELS││ │  │ ││KERNELS││ │  │ ││KERNELS││ │        │  
│  │ │└───────┘│ │  │ │└───────┘│ │  │ │└───────┘│ │        │  
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │        │  
│  └─────────────┘  └─────────────┘  └─────────────┘        │  
│           │               │               │                │  
│           └───────────────┼───────────────┘                │  
│    Batching, Scheduling, Load Balancing, Communication     │  
└─────────────────────────────────────────────────────────────┘  
```

**Flow:** Requests come in → System layer batches and distributes → Graph layer optimizes operations → Kernels execute on hardware → Results flow back up

### Why This Three-Layer View Matters

Running an LLM efficiently requires optimizing all three layers. You can't just write a fast kernel and call it a day. Here's why:

- The **System Layer** ensures your GPUs are fully utilized across a cluster and requests are batched efficiently.
- The **Graph Layer** streamlines your model's operations and eliminates wasteful computations.
- The **Kernel Layer** squeezes every drop of performance from the hardware.

Miss any layer, and your system will have bottlenecks. It's like having a Ferrari engine in a horse-drawn carriage—the potential is there, but the system design limits performance.

Our challenge starts in the kernel layer, where we'll learn to write high-performance GPU code. This high-level map gives us context for where our work fits in the bigger picture.

### Optimization Trade-offs Across Layers

Understanding the trade-offs helps identify where to focus optimization efforts:

| Constraint | Kernel Layer | Graph Layer | System Layer |
|------------|--------------|-------------|--------------|
| **Compute Bound** | Optimize arithmetic intensity | Use quantization, sparsity | Choose compute-optimal parallelism |
| **Memory Bound** | Improve data locality, fusion | Graph optimization, operator fusion | Use memory-efficient parallelism |
| **Communication Bound** | N/A (single GPU) | Minimize intermediate tensors | Optimize parallelism strategy |
| **Latency Critical** | Reduce kernel launch overhead | JIT compilation, CUDA graphs | Tensor parallelism, speculative decoding |
| **Throughput Critical** | Maximize occupancy | Batch operations | Data parallelism, continuous batching |

### Common Bottleneck Patterns

Recognizing these patterns helps diagnose performance issues:

| Symptom | Likely Layer | Common Causes | Solutions |
|---------|--------------|---------------|-----------|
| Low GPU utilization (<50%) | Kernel | Poor memory access, low arithmetic intensity | Optimize memory patterns, use tensor cores |
| High GPU util, slow overall | Graph | Inefficient operators, poor fusion | Operator fusion, quantization |
| Good single GPU, poor scaling | System | Communication overhead, load imbalance | Better parallelism strategy, batching |
| High latency, good throughput | System | Poor request scheduling | Continuous batching, speculative decoding |

### Quiz: Where Does the Bottleneck Live?

Let's test your understanding with a quick scenario:

**Scenario:** You have a perfectly optimized matrix multiplication kernel (Kernel Layer) running at 90% of theoretical peak performance. Your model uses the latest graph optimizations (Graph Layer) with perfect operator fusion. But your overall system throughput is still terrible.

**Question:** Which layer is likely the bottleneck, and what might be wrong?

---
<details>
<summary><b>Click to see the solution</b></summary>

**Answer:** The bottleneck is likely in the **System Layer**.

Even with perfect kernels and graph optimization, you can still have system-level issues:
- **Poor Batching:** Your system might be processing requests one at a time instead of batching them efficiently.
- **GPU Idle Time:** GPUs might be waiting for data or for other GPUs to finish their work.
- **Communication Overhead:** In multi-GPU setups, time spent transferring data between GPUs can dominate.
- **Load Imbalance:** Some GPUs might finish their work much earlier than others, leaving them idle.

This is why frameworks like SGLang or vLLM focus heavily on system-level optimizations like continuous batching and efficient memory management.

</details>

---

### Advanced Concepts: Cross-Layer Optimizations

Some cutting-edge techniques span multiple layers:

| Technique | Layers Involved | Description |
|-----------|-----------------|-------------|
| **Speculative Decoding** | Graph + System | Use small model to predict, verify with large model |
| **[MegaKernels](https://github.com/HazyResearch/Megakernels)** | Kernel + Graph | Fuse entire transformer blocks into single kernels |
| **Algorithm-Hardware Co-design** | All layers | Design algorithms specifically for hardware constraints |
| **Dataflow Architectures** | All layers | Execute operations when data becomes available |
| **[Prefill-Decode Disaggregation](https://arxiv.org/pdf/2401.09670)** | System | Separates prefill and decode phases onto different hardware for optimized latency and resource allocation |
| **[Flash Attention](https://arxiv.org/pdf/2205.14135)** | Kernel | Optimized attention mechanism with reduced memory bandwidth and improved efficiency for long sequences |



### What's Next

Tomorrow, we'll roll up our sleeves and dive into the kernel layer. We'll write our first CUDA kernel and see how low-level optimizations can dramatically impact performance. Expect hands-on GPU programming, memory access patterns, and practical examples that build on our GPU monster analogy.

The foundation is set. Now let's start building.

### Suggested Readings

1. **[Huizi Mao, "Understanding LLM System with 3-layer Abstraction"](https://ralphmao.github.io/ML-software-system/)**: A concise breakdown of LLM system architecture that inspired this framework.
2. **[Hao Zhang et al., "CSE-234: Data Systems for Machine Learning" Lecture Notes](https://hao-ai-lab.github.io/cse234-w25/assets/scribe_notes/jan9_scribe.pdf)**: Detailed insights into ML system design and optimization strategies.

---

**Special Thanks:** A huge shoutout to my friend Chun-Mao ([Michael](https://www.mecoli.net/)) Lai at UCSD for introducing me to the amazing and well-structured UCSD CSE234 course.