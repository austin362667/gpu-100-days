# Day 7: RL in LLM Post-training

Welcome to Day 7 of the GPU Challenge!

> Before diving into the RL framework, I’d like to introduce some basic concepts in RL and LLMs.

## RL Is Shaping the Way LLMs Think

The initial training of Large Language Models (LLMs) on massive datasets has been a cornerstone of their development. However, the next evolutionary leap is being driven by a powerful post-training technique: **Reinforcement Learning (RL)**. This approach is proving to be a game-changer, pushing LLMs beyond simply mimicking training data and into genuine reasoning and problem-solving capabilities.

## Why Shift Focus to RL?

In the early days of the LLM boom, models were primarily refined using **Supervised Fine-Tuning (SFT)**, where they learned to replicate "correct" answers from curated datasets. While effective, this approach still leaves room for improvement in LLMs.

Reinforcement Learning offers a more dynamic and adaptive learning process. In this framework:
- The LLM acts as an **"agent"**.
- It generates responses that are evaluated and assigned quantified **"rewards"** based on quality and preference.
- This feedback loop enables progressive improvement in problem-solving abilities.

## Key Advantages of RL in LLMs

- **Breaking the Data Wall**: RL generates vast amounts of training data, effectively addressing the "data wall" researchers face in pre-training. As the model interacts with its environment and receives feedback, it continuously creates new and diversified learning examples.

- **Enhanced Reasoning**: This approach significantly improves reasoning capabilities, particularly for well-defined intellectual tasks with verifiable answers like coding quizzes(e.g., [LeetCode](https://en.wikipedia.org/wiki/LeetCode)) or standardized math tests(e.g., [AIME](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination)).

## How RL Works with LLMs

The application of RL to LLMs involves three key stages:

### 1. The Rollout Phase
The LLM generates one or more responses to a given prompt.

### 2. The Scoring Phase
Generated responses are evaluated and assigned scores through methods that may include, but are not limited to:

#### 1. **Reinforcement Learning from Human Feedback (RLHF)**
Human labellers rank different model-generated responses.

#### 2. **Rule-based Verifiers**: 
For tasks with clear right/wrong answers (math, coding), predefined rules assess response correctness.

#### 3. **Reward Models**: 
For more open-ended tasks, a separate reward model is used to predict response quality. This model may be another generative model, as it has an effect equivalent to that of a regression model.

### 3. The Learning Phase
Scores are used to update the LLM's parameters using algorithms like [Proximal Policy Optimization (PPO)](https://openai.com/index/openai-baselines-ppo/), encouraging the production of more desired outputs.

### Mid-training and Cold-start Strategies

To make LLM thinks, we need it keeps yapping!

However, making models generate longer, more detailed responses isn't straightforward. It requires specialized training phases:

**Cold-start/Mid-training**: This intermediate phase encourages models to produce more verbose outputs, creating a foundation for subsequent RL training. During this phase, researchers focus on increasing the model's propensity to "think out loud."

### How to filter out and amplify the good signals?

**Oversampling**: Generate thousands (or as many as possible) of responses to the same input prompt and select the best ones—for example, by "highest score" or "self-consistency" (aka "majority voting"). This approach scales pretty well because:

- Incorrect intermediate steps may not significantly impact final accuracy.
- Step-wise verification is challenging to define and implement.
- End-to-end optimization (focusing on final answer correctness) may be more practical than **Process Reward Models (PRM)**.
- **Monte Carlo Tree Search (MCTS)** can be inefficient and difficult to scale.

Analyzing response distributions provides valuable insights into model behavior and improvement opportunities.

## The Power of Verifiability: The AI Version of [NP](https://en.wikipedia.org/wiki/NP_(complexity))

### "If you have the target, you can optimize"
A critical success factor for RL implementation is the ability to verify output quality. The **"asymmetry of verification"** principle states that for many tasks, checking a solution is far simpler than generating it. Examples:
- Verifying a Sudoku solution vs. solving the puzzle from scratch
- Checking mathematical proof correctness vs. discovering the proof
- Validating code functionality vs. writing the code

This asymmetry is why RL has been particularly successful in domains with objective success criteria.

### Structured Reward Systems

For open-ended tasks where defining good reward functions is challenging, innovative approaches have emerged:

**Rubrics as Rewards (RaR)**: A framework using structured, checklist-style rubrics as reward signals. These rubrics can be applied by other LLMs acting as "judges," providing nuanced feedback.

### The Flywheel: RL Helping RL

An exciting aspect of this field is the **symbiotic relationship** between RL and LLMs. As models become more capable through RL, they can improve the RL process itself:

- More advanced LLMs serve as more sophisticated "judges"
- Improved models provide more accurate and nuanced feedback
- This accelerates the improvement of subsequent model generations
- Creates a positive feedback loop driving rapid advancement

Just as we scale reasoning models, the improved models can serve as better verifiers for future training cycles.

## Rational Optimism for the Future

The progress in RL for LLMs provides grounds for rational optimism. Like previous deep learning advances, [LLM development follows S-curves](https://x.com/karpathy/status/1944435412489171119). When apparent limits are reached, brilliant minds in the field discover new approaches to transcend those barriers.

The journey toward more capable AI systems is far from over. By embracing reinforcement learning's power, we are unlocking new levels of performance and reasoning in large language models. The key is persistent effort. 🙂

---

## Suggested Readings

- [Thinking in Large Language Models](https://lilianweng.github.io/posts/2025-05-01-thinking/) - A high signal-to-noise blog with thoroughly formulated math equations regarding advanced ML researches by Lilian Weng at OpenAI. Good for people prefer paper-style explanations.

- [Asymmetry of Verification and Verifiers Law](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) - A valid manifesto by Jason Wei, GOAT AI researcher at OpenAI, famous for works of `o1` and `deep research`.

- [Scaling Reinforcement Learning Environments](https://semianalysis.com/2025/06/08/scaling-reinforcement-learning-environments-reward-hacking-agents-scaling-data/) - By emiAnalysis, the world's best, and diligently-run newsletter of SOTA AI research and AI infra led by Dylan Patel.