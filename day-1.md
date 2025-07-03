# Day 1: Your GPU is a Monster. Don't Let It Starve

Welcome to Day 1 of the GPU Challenge! Let's talk about how to tame the powerful GPU beast.


<div align="center">
<img src="assets/day-1-0.jpeg" alt="day-1-0.jpeg" width="200"/>
</div>


My journey into GPU programming started with a classic mistake. I'd run my code, summon `nvidia-smi`, and see `gpu-util` at 100%. "Success!" I thought. "My GPU is working hard." But my code was still slow. Why?

I was underestimating GPU's abilities and failed to unlock their full potential. It's more helpful to think of it as a monster. And I was letting it starve. To be more specific, `gpu-util` at 100% only means the monster is *awake*. It doesn't mean it's *feasting*. It could be awake but just nibbling on tiny snacks, leaving its immense digestive power completely untapped.

To truly unleash the beast, we need to understand how it eats. The **Roofline Model** is our guide to monster nutrition.

Think of your GPU monster as having two key attributes:

1.  **Its Mouth Size (Memory Bandwidth):** The size of its mouth determines how fast it can devour data, measured in **Bytes per Second**. This is the **Memory Roof**.
2.  **Its Stomach Digestion Speed (Compute Power):** The speed of its digestion determines how much useful work it can do, measured in **Floating Point Operations per Second (FLOP/s)**. This is the **Compute Roof**.

The key to feeding it effectively is the **caloric density** of the food you provide—its **Operational Intensity** or **Arithmetic Intensity** (`FLOPs / Byte`).

*   **Low-Density Food (Memory-Bound):** Operations like element-wise addition are like celery. The monster fills its mouth but gets very little energy. It's limited by how fast it can chew (memory bandwidth), while its powerful stomach sits idle.
*   **High-Density Food (Compute-Bound):** Operations like matrix multiplication are like pure protein. A single mouthful provides a massive amount of energy, letting the stomach digest at full capacity. Now, the monster is truly productive.

Our goal is to feed our monster the most calorically dense data-food we can, so it's limited by its digestion speed, not its mouth size.

### Math Quiz: What Should We Feed the Monster?

Let's get specific. Imagine our monster GPU has the following specs (similar to an NVIDIA A100):
*   **Digestion Speed (Peak Compute)**: 312 TFLOP/s (trillion fp16 operations/sec)
*   **Mouth Size (Peak Memory Bandwidth)**: 1.5 TB/s (trillion bytes/sec)

We want to feed it two different meals, both using a `4096x4096` matrix of fp16 numbers (2 bytes per number).

**Question 1: What is the caloric density (Operational Intensity in FLOPs/Byte) of...**
a) An element-wise addition (`D = A + B`)?
b) A matrix multiplication (`E = A @ C`)?

**Question 2: Based on the Roofline model, what is the monster's maximum *expected* digestion rate (in TFLOP/s) for each meal?**

---
<details>
<summary><b>Click to see the solution</b></summary>

**Solution 1: Caloric Density (Operational Intensity)**

First, let N=4096. A `4096x4096` matrix has `N*N` elements. In fp16, each matrix uses `N*N*2` bytes of memory.

*   **a) Element-wise addition (`A + B`):**
    *   **Operations:** `N*N` additions.
    *   **Memory:** Read A (`N*N*2` bytes) + Read B (`N*N*2` bytes) + Write D (`N*N*2` bytes) = `6*N*N` bytes.
    *   **Intensity:** `(N*N) / (6*N*N)` = **~0.167 FLOPs/Byte**. This is like water. Very low density.

*   **b) Matrix multiplication (`A @ C`):**
    *   **Operations:** Roughly `2*N*N*N` operations (a multiply and an add for each element in the inner loop).
    *   **Memory:** Read A (`N*N*2` bytes) + Read C (`N*N*2` bytes) + Write E (`N*N*2` bytes) = `6*N*N` bytes.
    *   **Intensity:** `(2*N^3) / (6*N^2)` = `N / 3` = `4096 / 3` = **~1365 FLOPs/Byte**. This is a power bar. Very high density.

**Solution 2: Expected Digestion Rate (Performance)**

*   **a) Element-wise addition:** With a density of 0.167, the monster is limited by its mouth size (memory-bound).
    *   Expected Performance = Mouth Size × Density = `1.5 TB/s * 0.167 FLOPs/Byte` = **~0.25 TFLOP/s**.
    *   The monster will be chewing as fast as it can but getting almost no work done.

*   **b) Matrix multiplication:** With a density of ~1365, is the food rich enough? Let's find the monster's "fullness point" (`Digestion Speed / Mouth Size`).
    *   Fullness Point = `312 TFLOP/s / 1.5 TB/s` = `208 FLOPs/Byte`.
    *   Since `1365 > 208`, our matmul is calorically dense enough to be compute-bound.
    *   Expected Performance ≈ **312 TFLOP/s**. The monster will be digesting at its absolute maximum speed.

</details>

---

### Toy Code Sample

Now, let's feed the beast and see if our predictions hold up. Run the code below and compare the "Achieved GFLOPS" with the answers from our quiz.

```python
import torch
import time

# Make sure you have a GPU
if not torch.cuda.is_available():
    print("CUDA is not available. This demo requires a GPU.")
    exit()

device = torch.device("cuda")
dtype = torch.float16 # Use float16 for Tensor Core goodness

# Let's define a large tensor size
# We'll use a 4096x4096 matrix
N = 4096
A = torch.randn(N, N, device=device, dtype=dtype)
B = torch.randn(N, N, device=device, dtype=dtype)
C = torch.randn(N, N, device=device, dtype=dtype)

# --- 1. Memory-Bound Meal: Element-wise Addition ---
# Operational Intensity is very low (~0.167 FLOPs/Byte).
# Performance should be limited by the monster's mouth (memory bandwidth).
torch.cuda.synchronize() # Wait for all previous work to finish
start_time = time.time()
D = A + B
torch.cuda.synchronize() # Wait for the operation to complete
end_time = time.time()

add_time = end_time - start_time
add_ops = N * N
add_gflops = (add_ops / 1e9) / add_time
print(f"Element-wise Add Time: {add_time:.6f} seconds")
print(f"Achieved GFLOPS (Add): {add_gflops:.2f}") # Compare this to your quiz answer! (0.25 TFLOP/s = 250 GFLOP/s)
print("-" * 20)


# --- 2. Compute-Bound Meal: Matrix Multiplication ---
# Operational Intensity is very high (~N/3 = 1365 FLOPs/Byte).
# Performance should be limited by the monster's stomach (compute power).
torch.cuda.synchronize()
start_time = time.time()
E = torch.matmul(A, C)
torch.cuda.synchronize()
end_time = time.time()

matmul_time = end_time - start_time
matmul_ops = 2 * N * N * N
matmul_gflops = (matmul_ops / 1e9) / matmul_time
print(f"Matrix Multiply Time: {matmul_time:.6f} seconds")
print(f"Achieved TFLOPS (MatMul): {matmul_gflops / 1000:.2f}") # Compare this to your quiz answer! (312 TFLOP/s)

```

### Suggested Readings

1.  [**"Making Deep Learning Go Brrrr From First Principles" by Horace He**](https://horace.io/brrr_intro.html): The main inspiration for this post. A must-read for a clear, first-principles-based mental model.
2.  **["Understanding GPU Performance: A Guide for Developers" by Arthur Chiao](https://arthurchiao.art/blog/understanding-gpu-performance/)**: The crucial article that explains why `gpu-util` can be misleading and why we need to think deeper, like a monster nutritionist.
3.  **[The Roofline Model (from the "JAX Scaling Book")](https://jax-ml.github.io/scaling-book/roofline/)**: An excellent and concise explanation of the Roofline model—the science behind feeding our monster effectively.