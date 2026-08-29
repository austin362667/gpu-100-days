---
marp: true
theme: default
paginate: true
class: lead
---

# Introduction to Multi-GPU Programming

**Why Use Multiple GPUs?**

  - **Strong Scaling:** Solve the same problem faster by using more computational resources.
      - *Example:* Weather forecasting that needs to be delivered on time.
  - **Weak Scaling:** Solve larger problems within the same time frame, or access more memory and resources for bigger computations.
  - **Efficient System Utilization:** Modern large-scale systems inherently feature multiple GPUs, requiring parallel code for efficient use.

---

```
Strong Scaling 🚀
+--------+      +-----+-----+
|        |      | GPU | GPU |
| Task A |  ->  +-----+-----+
|        |      | GPU | GPU |
+--------+      +-----+-----+
(Single GPU)    (Multi-GPU)

Weak Scaling 🌍
+--------+      +-------------+
| Task A |      | Task A | Task B |
| (Size X) |  ->  +-------------+
|        |      | Task C | Task D |
+--------+      +-------------+
(Single GPU)    (Multi-GPU, Size 4X)
```

-----

# Hardware Networking for AI Clusters - Node Level

**The Building Blocks of an AI Cluster**

  - **GPU-NIC Pair:** Each GPU is paired with its own NIC for external communication.
  - **Node Architecture:** Typically houses 8 GPU-NIC pairs, plus a CPU and memory.
  - **Scale-Up Technologies (Within Node):**
      - **NVLink:** High-speed GPU-to-GPU interconnect.
          - Faster than PCIe, bypasses CPU.
          - Enables **GPU Direct P2P transfers**.

---

```
            +------------------+
            |       CPU        |
            +--------+---------+
                     | PCIe
+-----------+--------+---------+-----------+
|           |                  |           |
|  +-------+--+     +-------+--+     +-------+--+ ... (8x)
|  |   GPU    |<--->|   GPU    |<--->|   GPU    |
|  +----------+     +----------+     +----------+
|      | NVLink         | NVLink         |
|  +-------+--+     +-------+--+     +-------+--+
|  |   NIC    |     |   NIC    |     |   NIC    |
|  +----------+     +----------+     +----------+
|                                              |
+----------------------------------------------+
                  Server Node
```

-----

# Hardware Networking for AI Clusters - Cluster Level

**Scaling Out to Larger Systems**

  - **Rack Structure:**
      - Two nodes per rack, connected via **Top of Rack (ToR) switch**.
      - 16 NICs for two 8-GPU nodes.
  - **Pod Structure:**
      - A collection of racks (256–8,000+ GPUs).
      - Can span buildings or geographies with data center interconnects.
  - **Interconnect Technologies:**
      - **RDMA:** Low-latency, CPU-bypassed memory transfers.
      - **InfiniBand:** Lossless, robust but limited ecosystem.
      - **RoCE/Rocky V2:** RDMA over Ethernet/IP.
      - **Emerging Standards:** Ultra Ethernet Consortium, TTPOe.

---

```
                 +--------------+
                 | ToR Switch   |
                 +------+-------+
                        |
       +----------------+----------------+
       |                                |
+------+---------+             +------+---------+
| Node 1 (8 GPUs)|             | Node 2 (8 GPUs)|
+----------------+             +----------------+
      Rack 1                           ... Racks -> Pod

```

-----

# Distributed Programming Models - Overview

**The Software Layer for Multi-GPU**

  - **Core Principle:** One process per GPU → simpler memory management & direct communication.
  - **Three Key Libraries:**
    1.  **MPI** – Message Passing Interface
    2.  **NCCL** – NVIDIA Collective Communications Library
    3.  **NVSHMEM** – NVIDIA SHared MEMory (PGAS model)

---

```
  +-----------+   +-----------+   +-----------+
  | Process 0 |   | Process 1 |   | Process 2 |
  +-----+-----+   +-----+-----+   +-----+-----+
        |               |               |
  +-----v-----+   +-----v-----+   +-----v-----+
  |   GPU 0   |   |   GPU 1   |   |   GPU 2   |
  +-----------+   +-----------+   +-----------+
```

-----

# Distributed Programming Models - MPI

**Message Passing Interface (MPI)**

  - **Definition:** Standard for process-to-process data exchange.
  - **Communication Model:**
      - **Two-Sided:** Explicit `send` / `recv`.
      - Collectives: `all_reduce`, `reduce`, `broadcast`.
  - **Key Characteristics:**
      - **Portable:** Works across diverse systems.
      - **CUDA-Aware:** Can transfer GPU buffers via **GPU Direct RDMA**.
      - **Overheads:** Needs stream sync if not stream-aware, possible host copies.
      - **Strong for structured scientific codes.**

---

```
Two-Sided Communication 🤝

Process 0 (GPU 0)               Process 1 (GPU 1)
+----------------+ send(data)   +----------------+
|                |------------->|                |
|                |<-------------|                |
+----------------+   recv(data) +----------------+
```

-----

# Distributed Programming Models - NCCL

**NVIDIA Collective Communications Library**

  - **Definition:** Optimized for GPU collectives (`all_reduce`, `all_gather`, `reduce_scatter`).
  - **Communication Model:**
      - Collectives, executed as **GPU kernels** for efficiency.
  - **Key Characteristics:**
      - **GPU-Centric:** Leverages GPU hardware & topology (NVLink).
      - **Stream-Aware:** Overlap comm & compute.
      - **Group Ops:** Reduce launch overheads.
      - **Wide Adoption:** Integrated into PyTorch, TensorFlow, etc.
      - **Auto-Optimized:** Chooses best protocol per topology/size.
      - **Roadmap:** Adding one-sided comm & symmetric memory.

---

```
NCCL All-Reduce 🔄

GPU 0      GPU 1      GPU 2      GPU 3
[d0]       [d1]       [d2]       [d3]
|          |          |          |
\+----------+----------+----------+
|
GPU Kernels
(Optimized Collective)
|
\+----------+----------+----------+
|          |          |          |
[sum(d)]   [sum(d)]   [sum(d)]   [sum(d)]

```

-----

# Distributed Programming Models - NVSHMEM

**NVIDIA SHared MEMory (NVSHMEM)**

  - **Definition:** GPU extension of PGAS (OpenSHMEM).
  - **Communication Model:**
      - **One-Sided:** `put`/`get` without matching receives.
      - **Symmetric Heap:** `nvshmem_malloc` memory shared across PEs.
  - **Key Characteristics:**
      - **Device-Initiated:** Comm from inside CUDA kernels (thread/warp/block).
      - **Direct NVLink Access:** Remote memory as local pointers.
      - **Simplifies Algorithms:** Suited for unstructured data & expert parallelism.
      - **Latency Optimized:** Fine-grained comm.
      - **Composability:** Works with MPI for setup.
      - **Restrictions:** Symmetric heap may limit heterogeneous use cases.

---

```

One-Sided Communication (put) ✍️

GPU 0 Kernel                     GPU 1 Memory
\+----------------+               +----------------+
| thread 0       |  put(data)    |                |
|   ...          |--------------\>| Symmetric Heap |
| thread n       |               |                |
\+----------------+               +----------------+
(No recv needed on GPU 1)

```

-----

# Choosing the Right Tool & Interoperability

**Making Informed Decisions**

  - **MPI:**
      - Best for CPU-MPI legacy codes, portable.
      - Watch out for sync overheads.
  - **NCCL:**
      - Best for deep learning workloads.
      - GPU-centric, stream-aware, widely supported.
  - **NVSHMEM:**
      - Best for one-sided, in-kernel comm, kernel fusion, NVLink access.
      - Restrictions with symmetric heap model.

---

**Interoperability:**

  - Common to **mix libraries**: e.g., MPI for setup + NCCL/NVSHMEM for GPU comm.
  - **Future:**
      - NCCL adding one-sided + symmetric memory.
      - MPI improving stream integration.
      - Increasing feature convergence across libraries.

---

```
    +--------------------------------+
    |              MPI               |
    |  (Portability, Legacy Code)    |
    +-----------------+--------------+
                      | Interoperability
    +-----------------v--------------+
    |             NCCL               |
    | (Deep Learning, Collectives)   |
    +-----------------+--------------+
                      | Interoperability
    +-----------------v--------------+
    |            NVSHMEM             |
    | (One-Sided, In-Kernel Comm)    |
    +--------------------------------+
```