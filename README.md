# MLSys Learning Notes

### Learn You a MLSys for Great Good (in public)!

### GPU Kernels

- [Your GPU is a Monster. Don't Let It Starve](./gpu-kernels/your-gpu-is-a-monster-dont-let-it-starve.md) -> Max out every part of the GPU (a napkin math first)
- [A High-Level Overview of LLM Systems](./gpu-kernels/a-high-level-overview-of-llm-systems.md) -> A broad overview without getting lost in technical details
- [Writing Your First CUDA Kernel](./gpu-kernels/writing-your-first-cuda-kernel.md) -> Introduction to GPU programming with a simple CUDA kernel
- [The Art of Pointer Arithmetic](./gpu-kernels/the-art-of-pointer-arithmetic.md) -> The underlying memory layout of tensor representation
- [Tiling and Shared Memory](./gpu-kernels/tiling-and-shared-memory.md) -> Dividing the matrix into blocks that fit within the cache
- [Global Memory Coalescing](./gpu-kernels/global-memory-coalescing.md) -> Combining adjacent accesses into single memory transaction
- [Learning CuTe DSL](./gpu-kernels/learning-cute-dsl.md)
- [CuTe DSL Learning Checklist](./gpu-kernels/cute-dsl-learning-checklist.md)
- [Introduction to Multi-GPU Programming](./gpu-kernels/introduction-to-multi-gpu-programming.md)

### RL Training Frameworks

- [RL in LLM Post-training](./rl-training-frameworks/rl-in-llm-post-training.md) -> This is the way LLMs can do reasoning
- [RL Framework Design Space](./rl-training-frameworks/rl-framework-design-space.md) -> Discuss RL infra form factor
- [A Log of Setting Up Slime](./rl-training-frameworks/a-log-of-setting-up-slime.md) -> The logs of me playing [Slime](https://github.com/THUDM/slime)
- [VeRL Q&A Notes](./rl-training-frameworks/verl-q-and-a-notes.md) -> A write-up of Haibin Lin’s introduction and Q&A on [VeRL](https://github.com/volcengine/verl) at PyTorch Webinar

### Low-precision Data Type

- [Don't Just `.cast()`](./low-precision-data-type/dont-just-cast.md) -> Notes on MXFP8, MXFP4, and NVFP4 formats and applications in PyTorch
- [The Missing 10 Bits](./low-precision-data-type/the-missing-10-bits.md) -> Notes on tensor-core precision
- [Different Precision for Different Workloads](./low-precision-data-type/different-precision-for-different-workloads.md)

### Speculative Decoding

- [Where Does the Speed-up Come From in Speculative Decoding?](./speculative-decoding/where-does-the-speed-up-come-from-in-speculative-decoding.md)

### On-policy Distillation

> WIP..

[Multi-Teachr OPD](https://yumoxu.notion.site/multi-teacher-on-policy-distillation)

[OPD Video by Jia-Bin Huang](https://www.youtube.com/watch?v=YH0YXgDWZXA)



### Model Taste

> WIP..
> I like Karina's writing style very much.
- [Everyone Is Looking at Meta's Glimmer-30B the Wrong Way
](https://substack.com/home/post/p-210598592)



