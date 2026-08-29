# Day 10: VeRL Q&A Notes

Welcome to Day 10 of our GPU Challenge!

Here’s a distilled, question‐by‐question list of my notes during the QA session of VeRL team sharing at PyTorch:

- Event Page: https://pytorch.org/event/verl-flexible-and-scalable-reinforcement-learning-library-for-llm-reasoning-and-tool-calling/

- Session Recording: https://www.youtube.com/watch?v=Vd79NmmqY3Q

- Project Github: https://github.com/volcengine/verl

> verl is the open-source version of [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2) paper.

1. **How efficient is the weight‐transfer implementation in VeRL, and is there room for improvement?**

   * Uses DTensor abstraction to re-shard model weights between generation and training.
   * Leverages NCCL (e.g., for `all-gather` ops) which auto-detects RDMA/InfiniBand under the hood.
   * Transfers happen per-parameter or bucketed to avoid OOM on large tensors.
   * Ongoing work with the vLLM team to expose more efficient weight-loading APIs and reduce “wasted” communication when tensor-parallel shards are partially unused.

2. **Why does VeRL depend heavily on Ray, and is there a plan for a simpler, non-Ray alternative?**

   * Ray is used as a Pythonic RPC engine to dispatch work to multi-controller groups.
   * Ray’s scheduling/resource‐management features aren’t strictly required. It mostly valued for ease of onboarding and good docs.
   * Torch RPC was explored but not upstream‐maintained, leaving few drop-in alternatives.
   * A lightweight custom RPC or revived Torch RPC could serve future non-Ray builds.

3. **Why prefer token‐in/token‐out over chat‐completion APIs for agent-RL?**

   * Token IDs avoid re-tokenization inconsistencies across multi-turn interactions.
   * Fine-grained control enables per‐token masks, sanity checks, and regression testing.
   * Ensures rollout and training see identical input/output representations.

4. **How do you see the role of Supervised Fine-Tuning (SFT) evolving given the rise of post-training RL?**

   * SFT remains cost-effective when high‐quality labeled data is available.
   * RL shines when verifiers or reward models can cheaply validate outputs.
   * Future “generalized verifiers” may further tip the balance toward RL for complex tasks.

5. **What’s the maximum context size you can train with VeRL, and is that sufficient?**

   * No formal benchmarking yet on max sequence length.
   * Relies on Megatron/FSDP backend to scale context via model parallelism.
   * Likely adequate for typical RL horizons, but exact limits depend on cluster size and memory budgets.

6. **Is visual-LLM support available in VeRL? How about other modalities?**

   * Vision(image and video) support is in progress; audio is forthcoming.
   * Integrating disparate encoders (e.g., Qwen’s special vision encoder) remains challenging.
   * Plans to standardize multimodal interfaces in a future modular design.

7. **How are you using RDMA for weight transfer from the trainer to the vLLM generator? Do you invoke `ibverbs` directly?**

   * Simply calls PyTorch’s `distributed.tensor.all_gather` (NCCL) which handles RDMA automatically.
   * Ray’s object store is being enhanced for GPU-direct RDMA, but VeRL itself delegates to NCCL.

8. **What’s the most difficult aspect of debugging RL training runs?**

   * High variance in rewards and entropy makes convergence noisy.
   * Tools that help:

     * Dumping rollout samples at each stage for manual inspection.
     * Logging log-probability differences between generation and training passes — if the differences are too large, something likely went wrong.
     * Unit tests for each component and daily end-to-end regression recipes.

9. **Follow-up: When using NCCL for RDMA weight transfer, do you use separate “multi-controller” actor groups? Are these RDMA transfers managed by a single controller?**

   * Each actor group sets up its own NCCL process group; no single controller bottleneck.
   * RDMA transfers are peer-to-peer within those groups, bypassing the central controller.

10. **You recently landed a one-step-off async training recipe. What technical challenges arose, and what future work do you envision?**

    * **System side:**

      * Managing stale parameters (K=1) without idling the device mesh.
      * Disaggregated trainer vs. generator placement to avoid CPU OOM during context switches.
    * **Algorithm side:**

      * Ensuring sample importance weighting or other importance-sampling tricks to correct for staleness.
      * Future: fully async pipelines with variable K-step staleness and interruptible inference servers to preempt long rollouts and maximize throughput.
