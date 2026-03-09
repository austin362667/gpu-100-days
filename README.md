# MLSys Learning Notes

### Learn You a MLSys for Great Good (in public)!

### GPU Kernels

- [Your GPU is a Monster. Don't Let It Starve](./day-1.md) -> Max out every part of the GPU (a napkin math first)

- [A High-Level Overview of LLM Systems](./day-2.md) -> A broad overview without getting lost in technical details

- [Writing Your First CUDA Kernel](./day-3.md) -> Introduction to GPU programming with a simple CUDA kernel

- [The Art of Pointer Arithmetic](./day-4.md) -> The underlying memory layout of tensor representiaobn

- [Tiling and Shared Memory](./day-5.md) ->  Dividing the matrix into blocks that fit within the cache

- [Global Memory Coalescing](./day-6.md) -> Combining adjacent accesses into single memory transaction

### RL Training Frameworks

- [RL in LLM Post-training](./day-7.md) -> This is the way LLMs can do reasoning

- [RL Framework Design Space](./day-8.md) -> Discuss RL infra form factor

- [Setting Up RL Infra](./day-9.md) -> The logs of me playing [Slime](https://github.com/THUDM/slime)

- [Notes from VeRL Talk](./day-10.md) -> 
A write-up of Haibin Lin’s introduction and Q&A on [VeRL](https://github.com/volcengine/verl) at PyTorch Webinar

### Low-precision Data Type 

- [Don't Just `.cast()`](./day-11.md) -> The note of learning from video: [
mxfp8, mxfp4, nvfp4 formats and applications in PyTorch - Vasily Kuznetsov & Driss Guessous, Meta](https://www.youtube.com/watch?v=Up0EfrudTSQ)

- [The Missing 10 Bits](./day-12.md) -> The note of learning from blog: [Some Matrix Multiplication Engines Are Not As Accurate As We Thought](https://pytorch.org/blog/some-matrix-multiplication-engines-are-not-as-accurate-as-we-thought/)

### Speculative Decoding

- [Speculative Decoding: Part 1](./day-15.md)