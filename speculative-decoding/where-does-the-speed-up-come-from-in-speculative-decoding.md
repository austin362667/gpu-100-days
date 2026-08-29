# Day 14: Where does the speed-up come from when applying Speculative Decoding?

Welcome to Day 14 of the GPU Challenge!

Today, I'm going to dive into a major technique to speed-up LLM inference with a  introduction, that is [**Speculative Decoding**](https://arxiv.org/abs/2211.17192).

Specifically looking under the hood at exactly *why* it speeds up inference and how it essentially tricks the GPU into doing what it does best.

---

### Memory-Bound Autoregressive Decoding

To understand the solution, we have to understand the problem. Standard autoregressive decoding generates text one token at a time. For *every single token*, the GPU must load the entire massive weight matrix of the target model (including the massive `lm_head`) from its High Bandwidth Memory (HBM) into its compute cores.

Because modern GPUs have immense mathematical compute power (FLOPs) but relatively slow memory transfer speeds, the compute cores spend most of their time sitting idle waiting for the weights to arrive. This operation is fundamentally a  heavily **memory-bound** Matrix-Vector multiplication.

### Speculative Decoding

Speculative decoding tackles this inefficiency by breaking generation into two steps:

1. **The Draft Phase:** A much smaller, faster model (the "draft model") quickly guesses a sequence of  tokens sequentially.
2. **The Verification Phase:** The large "target model" takes those  guessed tokens and verifies them in a single forward pass.

But where does the massive speed-up actually come from? The magic lies entirely in the Verification Phase.

### Think verification in target model like a "prefill" stage for draft tokens

The best way to understand the speed-up is to look at the **Prefill stage** of standard LLM inference. When you send a prompt to an LLM, it doesn't read the prompt one word at a time. It processes the entire prompt in parallel, computes the Key-Value (KV) cache for all tokens simultaneously, and outputs the first generated token. This is a highly efficient **compute-bound** Matrix-Matrix multiplication, meaning it fully utilizes the GPU's massive FLOPs in parallel.

Speculative decoding essentially forces the slow, sequential decode phase into a series of highly efficient **"prefills"** in single batch.

When the target model receives the  draft tokens, it treats them exactly like a user prompt:

1. It appends the  tokens to the existing sequence.
2. It does a parallel forward pass over all  tokens at once.
3. It computes the output distributions at the `lm_head` for  positions simultaneously.

By doing this, we pay the massive "shipping cost" of loading the model's weights from memory just *once*, but we get  tokens worth of math out of it. We successfully convert a memory-bound operation back into a compute-bound operation.

### With KV Cache

Since the target model treats these draft tokens like a prefill, it computes the Query (Q), Key (K), and Value (V) matrices for all  tokens in parallel using a causal attention mask.

As always, the Query matrices are are used to calculate attention scores and are immediately discarded. The Key and Value matrices stayed in memory instead

Once the target model computes the logits, it uses rejection sampling to verify the tokens:

* **For accepted tokens:** The newly computed K and V matrices are already sitting perfectly in the KV cache. We get these tokens practically "for free."
* **For rejected tokens:** If the draft model guessed 5 tokens, but the target model rejects token #3, it simply leaves the KV cache for tokens #1 and #2 intact. For tokens #3, #4, and #5, it just rolls back the sequence length pointer (a trick easily handled by systems like PagedAttention). The GPU simply pretends the cache ends at token #2 and overwrites the discarded data on the next pass.

### Wrap-up

Speculative decoding doesn't make the large model execute its math faster; it batches the workload. By using the draft model's guesses as a "prefill prompt", the target model avoids constantly waiting on memory transfers, massively increasing hardware efficiency.

---

### References & Further Reading

1. [**Google Research Blog: Looking back at speculative decoding (Dec 2024)**](https://research.google/blog/looking-back-at-speculative-decoding/) by Yaniv Leviathan, Matan Kalman, and Yossi Matias. *(A great retrospective on their foundational 2022 paper, ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/pdf/2211.17192).)*

2. [**High Performance LLM Inference in Production:**](https://www.youtube.com/watch?v=4gJGBEDbZp4&t=3211s) *Charles Frye at Modal explains inference architectures, specifically noting in the last section (starts from 3211s) how tuning draft models to specific token distributions (like math or agentic workloads) yields significant speedups.*

3. [**Faster LLMs: Accelerate Inference with Speculative Decoding:**](https://www.youtube.com/watch?v=VkWlLSTdHs8)
*An easy-to-understand video tutorial by IBM visually explaining the draft/verify mechanics.*

4. [**MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads:**](https://arxiv.org/pdf/2401.10774) Instead of a separate draft model, Medusa adds extra decoding heads directly to the original model. It predicts multiple future tokens simultaneously, achieving up to 3.6x speedups without touching the original target model's weights.

5. [**EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty:**](https://arxiv.org/pdf/2401.15077) This approach extrapolates hidden state features to predict draft tokens without fine-tuning the target model. It achieves similar speedups to Medusa but boasts incredibly high acceptance rates.