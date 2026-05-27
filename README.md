# The Gradient Descent Lottery

## How the 1986 Commitment to Euclidean Parameter Space Quietly Selected the Optimization Geometry of Every Subsequent Learning Machine, 1847–2026

---

> *"A research idea wins not because it is theoretically superior, but because it is perfectly matched to the available hardware and software ecosystem."*  
> — Sara Hooker, *The Hardware Lottery*, 2020

> *"The natural gradient method takes the steepest descent in the space of realizable distributions rather than the space of parameters, where the Riemannian metric is used for computing distance — a metric that approximates the square root of KL divergence locally and is only dependent on the distribution itself, not the parameterization."*  
> — Amari & Nagaoka, *Methods of Information Geometry*, 2000

> *"Adam is an approximation of natural gradient descent using the diagonal empirical Fisher information matrix — an approximation so aggressive it throws away all off-diagonal curvature information, all cross-parameter interactions, and the exact Fisher in favor of the empirical Fisher, then takes the square root of what remains."*  
> — Hwang, *FAdam*, 2024

> *"The data was curved. The substrate was flat. The optimizer was charged with moving through the former on the geometry of the latter. The bill has been paid in steps."*  
> — The Gradient Descent Lottery, 2026

---

## Abstract

The Hardware Lottery, as Sara Hooker named it in 2020, is the proposition that a research idea wins not because it is theoretically superior but because it is perfectly matched to the available hardware and software ecosystem. The Point Lottery Hypothesis (Ren, 2026) identified the prior scale at which the deepest co-design event in modern computing was decided: the 1985 IEEE 754 floating-point standardization, which institutionalized hidden iteration and selected for flat, non-iterative arithmetic substrates. *Attention Is All You Need to Sell Silicon* (Ren, 2026) identified the mid-scale event: the 2017 Transformer, co-designed with TPU v2 to maximize systolic-array matrix multiplication utilization. *The Matmul Ceiling* (Ren, 2026) identified the forward constraint: the economic wall that matmul-era biology compute hits by 2028, where the five architectural mismatches between biology workloads and GEMM-shaped silicon compound into board-level capital crises.

This document identifies the intervening scale — sitting between the 1985 arithmetic lottery and the 2017 silicon lottery, bridging both — at which the optimization geometry of the entire deep learning enterprise was decided.

The event is the 1986 popularization of backpropagation-with-gradient-descent (Rumelhart, Hinton, Williams, 1986) as the universal training algorithm, institutionalized in software, frameworks, and research practice through the following decade. The fight, compressed to its operational core, was between two ways of navigating a loss landscape on a finite machine. The first, natural gradient descent, follows the geodesic of the statistical manifold defined by the model's parameter family: it preconditions each gradient step by the inverse of the Fisher information matrix, making the update reparameterization-invariant, curvature-aware, and, near any local optimum, asymptotically efficient in the sense of Cramér-Rao. The second, stochastic gradient descent and its first-order variants (momentum SGD, Adam, AdaGrad, RMSProp), updates parameters by a scaled version of the raw Euclidean gradient — treating the parameter space as flat, ignoring all curvature, and imposing no geometric structure beyond the coordinate system the programmer happened to choose.

By 1998, when Amari formalized natural gradient descent in the language of information geometry, the theoretical case was complete. The natural gradient descent update is invariant to reparameterization; vanilla gradient descent is not. Near a local optimum, natural gradient descent converges in a number of steps that scales with the square root of the condition number; vanilla gradient descent converges in a number of steps that scales with the condition number itself (Shrestha, 2023). The Fisher information matrix is the correct Riemannian metric on the statistical manifold; the identity matrix, which vanilla gradient descent implicitly uses, is not. The theoretical verdict was not close.

The practical verdict was decided by hardware. The Fisher information matrix G for a model with N parameters is an N×N positive semi-definite matrix. Its storage is O(N²). Its inversion is O(N³). For a 150-million-parameter GPT-2, this is a 1.5×10¹⁶-element matrix — 120 petabytes of storage for a single float32 entry per element. For a 70-billion-parameter LLaMA-2, the storage is the cube root away from the heat death of any known data center. The computation is exact and impossible. Every practical optimizer in the history of deep learning is an approximation of natural gradient descent that trades off the richness of G for the feasibility of hardware. Adam uses the diagonal of the empirical Fisher — two orders of approximation from the theoretically correct object. K-FAC (Martens & Grosse, 2015) uses a Kronecker-factored block-diagonal approximation — closer, but still O(√N) rather than O(1) in storage. Shampoo (Gupta et al., 2018) uses full matrix preconditioners per layer tensor — closer still, but requiring eigendecompositions that are not natively supported on any production accelerator. Muon (Jordan, 2024) uses Newton-Schulz orthogonalization of Nesterov momentum — a spectral-norm steepest descent that approximates the geometry without naming it as such.

The **Gradient Descent Lottery** is the claim that this choice — natural gradient vs. Euclidean gradient, curvature-aware vs. curvature-blind, Riemannian vs. flat, Fisher-preconditioned vs. identity-preconditioned — quietly selected the optimization geometry of every subsequent learning machine. The 1986 commitment to gradient descent was not made because gradient descent is theoretically correct. It was made because gradient descent is hardware-correct: each parameter update is an element-wise scalar multiply, requiring O(N) storage and O(N) compute, natively supported on every ALU ever built. The Fisher-preconditioned update requires none of that. The substrate chose the optimizer; the optimizer shaped the architecture; the architecture shaped the research agenda; and the public has been paying in training steps, energy bills, and unrealized capability ever since.

This document is the genealogy of that choice. Section 1 establishes the two optimization geometries through a thought experiment. Section 2 traces the 1847–1986 fight, from Cauchy's gradient to Rumelhart's backpropagation. Section 3 characterizes the 1986 commitment and its institutional ossification through 2014. Section 4 inventories what was lost — the curvature coordinate. Section 5 reads the Fisher information matrix as the missing half of every modern optimizer. Section 6 reads the corpus's prior frameworks as partial recoveries of the lost curvature. Section 7 maps the three costs. Section 8 surveys the 2023–2026 second-order renaissance — Sophia, Muon, SOAP, FAdam, Distributed Shampoo — as the workload pushing back through the substrate's geometric amnesia. Section 9 forecasts the 2027–2030 optimization substrate. Section 10 states eight predictions. Section 11 closes.

---

## Table of Contents

1. [Two Geometries — A Thought Experiment](#1-two-geometries--a-thought-experiment)  
2. [The Long Fight, 1847–1986](#2-the-long-fight-18471986)  
3. [The 1986 Commitment](#3-the-1986-commitment)  
4. [What Was Lost: The Curvature Coordinate](#4-what-was-lost-the-curvature-coordinate)  
5. [The Fisher Information Matrix as the Missing Half](#5-the-fisher-information-matrix-as-the-missing-half)  
6. [Reading the Corpus Backward](#6-reading-the-corpus-backward)  
7. [The Three Costs](#7-the-three-costs)  
8. [The 2023–2026 Reversal](#8-the-20232026-reversal)  
9. [The Long Map](#9-the-long-map)  
10. [Predictions](#10-predictions)  
11. [The Manifold That Forgot — A Fourth Thought Experiment](#11-the-manifold-that-forgot--a-fourth-thought-experiment)  
12. [Closure](#12-closure)  
13. [Sources](#13-sources)

---

## 1. Two Geometries — A Thought Experiment

Place two researchers, in two different rooms, in front of the same task: train a two-layer neural network to fit a smooth target function on a hundred data points. Both researchers have unlimited compute. Both researchers have the correct loss function. They differ only in their optimizer.

The first researcher uses **stochastic gradient descent**. She initializes parameters θ ∈ ℝᴺ at random. At each step she computes the gradient ∇L(θ) of the loss with respect to parameters, multiplies by a learning rate η, and subtracts: θ ← θ − η∇L(θ). The update is a step in the direction of steepest descent in the *Euclidean* coordinate space she happens to be using. If she had parameterized the network differently — if she had replaced the weights by their logarithms, or by a rotation of them — the gradient direction would change. The Euclidean gradient is not intrinsic to the model. It is an artifact of parameterization. The learning rate must be tuned; the convergence speed depends on the condition number κ of the Hessian at the solution; if the loss landscape is highly anisotropic (as it is in virtually every trained neural network), convergence slows by a factor of κ, which in practice ranges from 10² to 10⁶. She will converge. It will take time.

The second researcher uses **natural gradient descent**. He initializes identically. At each step he computes not just the gradient but the Fisher information matrix: G(θ) = 𝔼_{x∼p(x)}[∇ log p(y|x;θ) ∇ log p(y|x;θ)ᵀ], the expected outer product of the log-likelihood gradient. He then updates: θ ← θ − η G(θ)⁻¹ ∇L(θ). This update moves in the direction of steepest descent in the *statistical manifold* of the model family, equipped with the Fisher-Rao metric. The update is invariant to reparameterization: if he had logged the weights, or rotated them, the natural gradient update would produce an identical trajectory in the space of distributions. Near the solution, natural gradient descent converges in O(√κ) steps; SGD converges in O(κ) steps (Shrestha, 2023; Martens, 2020). For κ = 10⁶, this is the difference between one thousand steps and one million steps. He will converge faster. Much faster.

What did the second researcher need that the first did not? He needed to compute and invert G(θ), an N×N matrix. On the 100-parameter toy problem, this is a 100×100 matrix — trivial. On a 100-million-parameter language model, this is a 10¹⁶-element matrix — impossible. On a 1-trillion-parameter frontier model, it is an object whose very description requires inventing new units. The Fisher information matrix is the correct object. The correct object cannot be computed. The incorrect object — the identity matrix, implicit in every vanilla gradient descent update — can always be computed, because it is already computed: it costs nothing to multiply by the identity.

This is the seed observation. Two optimization geometries existed. They differed not principally in their mathematical relationship to the problem — both can, in principle, find the same solution — but in their treatment of curvature. Natural gradient descent exposes and exploits the curvature; Euclidean gradient descent conceals it. The 1986 backpropagation paper institutionalized concealment. The institutional consequences, fully visible only forty years later, are the subject of this document.

---

## 2. The Long Fight, 1847–1986

The fight began in 1847 with Augustin-Louis Cauchy's paper *Méthode générale pour la résolution des systèmes d'équations simultanées*. Cauchy, working on iterative solution of systems of equations, proposed following the direction of steepest descent of a quadratic form — the negative gradient — as a general-purpose optimization strategy. The method was geometrically natural, computationally tractable with pencil-and-paper arithmetic, and theoretically justified for convex quadratics. It was also, for non-quadratic and ill-conditioned problems, potentially very slow: Cauchy already noted that convergence on highly anisotropic problems required many small steps. He had no practical machinery to fix this. The method entered the literature as the method of steepest descent.

The missing correction had a name by 1869: the Hessian matrix. If the gradient points in the direction of steepest descent in a flat (Euclidean) space, the curvature of the loss landscape is encoded in the second derivatives — the N×N matrix H(θ) = ∂²L/∂θᵢ∂θⱼ. Newton's method, which goes back to 1669 in its scalar form, applies the correction θ ← θ − H(θ)⁻¹ ∇L(θ), moving in the direction of steepest descent on the *quadratic approximation* of the loss rather than on the loss itself. Newton's method converges quadratically near a local minimum — one step doubles the number of accurate digits. It requires computing and inverting H(θ), an O(N²) storage and O(N³) computation for N parameters. For Cauchy's small systems, this was feasible.

For the large neural networks of the 1980s, it was not. The Hessian of a network with N = 10⁶ parameters is a 10¹²-element matrix. At four bytes per element, this is four terabytes — for the Hessian alone. The available RAM in 1986 was measured in megabytes. Newton's method was a non-starter.

The information-geometric understanding of what was actually wrong took longer to arrive. In 1945, C. R. Rao published *Information and the Accuracy Attainable in the Estimation of Statistical Parameters*, deriving the Cramér-Rao bound and identifying the Fisher information matrix as the natural metric on the space of statistical models: the metric that, for any smooth statistical model, defines the intrinsic distance between distributions. In 1945, numerical optimization and statistical theory were in separate communities, and the connection was not made.

In 1972, S.-I. Amari, working in Tokyo, began developing the field he would call *information geometry*: the application of Riemannian differential geometry to the space of probability distributions. His key observation, fully developed by the 1990s, was that the statistical manifold of a parametric model family is not flat — it has intrinsic curvature, encoded in the Fisher information matrix, that is independent of any choice of parameterization. Moving through this manifold requires respecting its geometry; the Euclidean gradient, which is parameterization-dependent, is the wrong tool.

Meanwhile, in the practical optimization community, the standard approach for large neural networks was evolving along a different track. Werbos (1974) had described backpropagation in his PhD thesis, but the influential formulation came from Rumelhart, Hinton, and Williams (1986) in *Learning Representations by Back-propagating Errors*. The paper showed that the gradient of a loss function with respect to the parameters of a layered network could be computed in a single backward pass through the network — the same computational cost as a forward pass — using the chain rule of calculus. This was the key algorithmic insight: gradients were *cheap*. A single forward-backward pass gave the exact gradient of the loss. Gradient descent on neural networks, which had seemed prohibitively expensive, was now feasible.

The gradient was cheap. The curvature was not. Rumelhart et al. trained with gradient descent. The community followed.

In 1998, Amari published *Natural Gradient Works Efficiently in Learning* in the journal *Neural Computation*. This paper is the direct predecessor of the modern natural gradient literature. Amari proved: (1) the natural gradient converges in a number of iterations that is independent of the condition number κ, asymptotically near a local minimum; (2) the natural gradient is the direction of steepest descent in the Riemannian manifold of the model's parameter family equipped with the Fisher-Rao metric; (3) the natural gradient is invariant to reparameterization; (4) the natural gradient reduces, for linear networks with Gaussian data, to the exact solution in one step. The theoretical case against Euclidean gradient descent was now complete, formal, and published in a top journal. The practical case for it — cheapness, hardware compatibility, framework support — was equally complete, equally formal, and enforced by every GPU vendor, every deep learning framework, and every benchmark leaderboard in existence.

The fight, in this distillation, lasted from 1847 to 1998. Gradient descent won in 1986, twelve years before the theoretical verdict was in. The commitment was made not on theoretical grounds but on practical ones: the gradient was cheap, the hardware was ready, and the Hessian and Fisher were not.

---

## 3. The 1986 Commitment

A single paper does not isolate the moment, but 1986 is the year the backpropagation-plus-gradient-descent combination became the consensus training algorithm for neural networks, through the *Nature* paper that brought it to the scientific mainstream. Three consequences of the 1986 commitment, played out over the following decade, fixed the optimization geometry of the next forty years.

**First: the framework.** The backpropagation algorithm is an algorithm for computing gradients. It is not an algorithm for computing Hessians or Fisher information matrices. Its computational graph — the chain of forward activations and backward error signals — terminates in first derivatives. The entire automatic differentiation infrastructure that grew from backpropagation (Theano, 2010; TensorFlow, 2016; PyTorch, 2017) is built to compute gradients efficiently and cheaply. Computing the full Fisher information matrix in any of these frameworks requires forming and backpropagating through O(N) separate gradient computations, at O(N) the cost of a single backward pass. The framework design encoded the 1986 commitment in software: gradient computation is first-class; curvature computation is an expensive afterthought.

**Second: the hardware.** The GPU acceleration of deep learning (Raina et al., 2009; Krizhevsky, Sutskever, Hinton, 2012) was acceleration of matrix multiplication. A forward pass through a neural network is a sequence of matrix multiplications (the linear layers) and element-wise operations (the nonlinearities). A backward pass is a sequence of matrix multiplications and element-wise operations in reverse. The gradient is the output of a sequence of matrix multiplications. The Fisher information matrix is the expected outer product of a vector — an N-dimensional vector whose outer product is an N×N matrix, requiring a rank-one update to an N×N accumulator per data point. On GPU hardware optimized for tall, thin matrix multiplications, this is deeply inefficient: the accumulation pattern of the Fisher outer product is entirely different from the access pattern of GEMM kernels. The hardware optimization that made gradient descent fast did not make Fisher computation fast, and the FLOP-counting model that followed (GFLOPS, then TFLOPS, then petaFLOPS) measured optimization cost in gradient-descent units. Fisher computation was, in this accounting, a luxury that could not pay for itself.

**Third: the architecture.** Between 1986 and 2017, the neural architectures that survived were architectures that made gradient descent work well. Batch normalization (Ioffe & Szegedy, 2015) normalizes activations to have zero mean and unit variance at each layer — a per-layer preconditioning operation that smooths the loss landscape and reduces the effective condition number seen by gradient descent. Layer normalization (Ba et al., 2016) does the same for sequence models. Residual connections (He et al., 2016) make the gradient flow through a deep network more like the gradient of a shallow one, reducing the effective condition number further. The entire architecture design program of 2014–2017 was, in retrospect, a program of making gradient descent work on harder and harder problems — of compensating, in software, for the curvature information that the optimizer was blind to. As the Riemannian manifold of a deep network is generically ill-conditioned, and gradient descent is agnostic to this ill-conditioning, the field spent a decade discovering architectural proxies for the preconditioning it could not afford to compute directly.

The 1986 commitment is not a mistake. For the problems of 1986 — small networks, limited data, constrained hardware — it was the correct choice, for the same reason the 1985 IEEE 754 floating-point standard was the correct choice for the workloads of 1985. The mistake, if there is one, is the forty-year failure to name the commitment as a commitment, to trace its consequences, and to ask when the workloads would outgrow it.

---

## 4. What Was Lost: The Curvature Coordinate

The 1986 commitment to gradient descent was, at the level of its consequences, a commitment to discard the curvature of the loss landscape from the optimizer's view. What follows is the inventory of what could not be expressed when the curvature was hidden.

**Reparameterization invariance.** A well-designed optimization algorithm should produce the same optimization trajectory regardless of how the model is parameterized. If a researcher multiplies all weights in a layer by 2 (which changes nothing about the model's function if the next layer's weights are halved), gradient descent will take a completely different path. Natural gradient descent will not: the natural gradient is intrinsically tied to the statistical manifold of the model family, not to the coordinate system the researcher chose (Martens, 2020; Amari, 1998; Ollivier, 2013). Every training run in the history of deep learning is, to a first approximation, a fight between the implicit geometry of the model's statistical manifold and the explicit assumption of flatness embedded in the optimizer. The optimizer loses, and the researcher compensates with learning rate schedules, warm-up phases, gradient clipping, and normalization layers.

**Condition-number-independent convergence.** Near a local minimum, SGD converges in O(κ) gradient steps, where κ is the condition number of the Hessian (ratio of largest to smallest eigenvalue). Natural gradient descent converges in O(√κ) steps (Amari, 1998; Shrestha, 2023). For deep language models, empirical estimates of κ in the final layers range from 10⁴ to 10⁷ (Ghorbani et al., 2019; Yao et al., 2020). A training run of 300,000 steps for a standard Transformer may be consuming 10–10,000× more steps than a curvature-aware optimizer would require. Sophia (Liu et al., 2023) provides direct empirical evidence: a diagonal Hessian preconditioner reduces GPT-2 pretraining steps by 50% with no change in architecture or data — a single recovery of a single dimension of curvature cuts training cost in half.

**Layer-heterogeneous preconditioning.** The curvature of a neural network's loss landscape is not homogeneous across layers. Early layers, which sit far from the output in the gradient graph, typically have very small curvature (flat directions, slow convergence). Final layers, which sit near the output, typically have large curvature (sharp directions, potential instability). A curvature-aware optimizer allocates step sizes proportional to the inverse curvature at each layer, effectively routing optimization budget toward directions that return the most loss decrease per step. Gradient descent with a single learning rate cannot do this. The compensating mechanism — learning rate schedulers, weight decay tuning, gradient clipping, per-layer learning rate multipliers — consumes tens of thousands of practitioner-hours per frontier training run and produces results that are recognized as heuristics rather than principled solutions.

**Cross-parameter interaction.** The Fisher information matrix G is a full N×N matrix. Its off-diagonal entries encode the degree to which a step in one parameter direction changes the effective gradient in another. In the attention mechanism of a Transformer, the query, key, and value projections are deeply coupled — the curvature of the loss with respect to W_Q is entangled with W_K through the attention pattern, which is a function of both. A diagonal optimizer (Adam, AdaGrad, RMSProp) treats these as independent, preconditions each with its own scalar, and misses all cross-parameter curvature. K-FAC (Martens & Grosse, 2015) recovers a Kronecker-factored approximation of the block-diagonal Fisher, capturing some of this structure. The evidence is consistent: K-FAC converges faster than Adam on many tasks (Martens, 2020). The field has not universally adopted K-FAC because its Kronecker inversion step is not natively supported on any production accelerator's instruction set.

**The optimizer–architecture coupling.** Because gradient descent is curvature-blind, and the architectures that emerged between 1986 and 2017 were designed to make gradient descent work, the architecture and the optimizer are coupled in ways that make it difficult to improve either independently. The Transformer, with its LayerNorm and residual connections, is partly a device for making gradient descent's ignorance of curvature less costly. A curvature-aware optimizer might not need LayerNorm; it might not need warmup schedules; it might not need gradient clipping. The architectural choices that survive in a gradient-descent regime are architectural choices that smooth the loss landscape — and smoothing the loss landscape for gradient descent is not the same as designing an architecture that is geometrically optimal for the underlying task.

What was lost, compressed to a phrase: the curvature coordinate. The 1986 commitment kept the gradient and threw away the landscape. For thirty years of shallow, narrow, well-conditioned networks, this trade looked nearly free. From 2017 forward, with models whose loss landscapes are deep, high-dimensional, highly non-isotropic, and entangled across billions of parameters, it has not been free — and the cost is being paid in training steps, energy, and in the research programs that never existed because the optimizer could not see the landscape they were navigating.

---

## 5. The Fisher Information Matrix as the Missing Half

Every optimizer that has ever trained a deep neural network can be located on a single axis: its distance from the natural gradient update G(θ)⁻¹ ∇L(θ).

**Stochastic gradient descent** (Robbins & Monro, 1951; Rumelhart et al., 1986) uses the identity matrix as preconditioner: the update is I⁻¹ ∇L = ∇L. Distance from natural gradient: maximum. Computational cost: O(N). Convergence: O(κ) steps.

**AdaGrad** (Duchi, Hazan, Singer, 2011) uses the diagonal of the cumulative sum of squared gradients as preconditioner. This is the time-averaged diagonal of the empirical outer product of gradients — a diagonal approximation to the empirical Fisher information matrix, averaged over the training trajectory. Distance from natural gradient: large, but non-trivially bounded below the identity. Computational cost: O(N). Convergence: better than SGD on sparse gradients.

**RMSProp** (Hinton, 2012, unpublished lecture notes) uses a running exponential average of the squared gradient — an exponentially decaying version of AdaGrad's preconditioner. Same diagonal approximation, with temporal discounting. Computational cost: O(N).

**Adam** (Kingma & Ba, 2014) combines the first moment (exponential average of gradient) and the second moment (exponential average of squared gradient) as preconditioner. The denominator in Adam's update — √(v_t) + ε, where v_t is the running average of g² — is the square root of the diagonal empirical Fisher information matrix. FAdam (Hwang, 2024) makes this precise: Adam is a natural gradient optimizer using the diagonal empirical Fisher information, with two approximation layers compounding each other. The first approximation: diagonal (all off-diagonal Fisher entries dropped). The second approximation: empirical Fisher (the expected outer product is replaced by the gradient at a single batch point, avoiding the need to marginalize over the model's predictive distribution). Both approximations are critical and together reduce the N×N information-geometric object to an N-vector. Computational cost: O(N). Convergence: better than SGD on ill-conditioned problems; still O(κ) in the worst case.

**K-FAC** (Kronecker-Factored Approximate Curvature; Martens & Grosse, 2015) approximates the block-diagonal Fisher matrix for each layer as a Kronecker product of two smaller matrices: G_l ≈ A_l ⊗ S_l, where A_l is the input covariance at layer l and S_l is the output gradient covariance at layer l. The inversion factorizes: (A_l ⊗ S_l)⁻¹ = A_l⁻¹ ⊗ S_l⁻¹, reducing the O(d_l⁴) Kronecker block inversion to two O(d_l³) matrix inversions. This is a genuine recovery of block-off-diagonal Fisher structure — it captures the coupling between input and output directions in each layer's weight matrix. K-FAC convergence on language models is faster than Adam's (Martens et al., 2018). K-FAC's per-step overhead is 1.2–2× that of Adam (Shrestha, 2023). The reason K-FAC is not universal: the Kronecker inversion step is a general matrix factorization, not a GEMM, and general matrix factorization does not map efficiently onto systolic-array hardware. Computational cost: O(N^{3/2}) amortized. Convergence: O(√κ) on block-diagonal tasks; still approximation-limited otherwise.

**EKFAC** (Eigenvalue-Corrected K-FAC; George et al., 2018) adds eigenvalue corrections to the K-FAC approximation, improving accuracy at modest additional cost. Still not universal deployment because of the eigendecomposition requirement.

**Shampoo** (Gupta, Kale, Kumar, Rajagopalan, Shamir, 2018; Anil et al., 2020; Vyas et al., 2024) abandons the diagonal altogether and uses full matrix preconditioners per layer tensor dimension. For a weight matrix W ∈ ℝ^{m×n}, Shampoo maintains two preconditioner matrices L ∈ ℝ^{m×m} and R ∈ ℝ^{n×n} and applies the update W ← W − η L^{-1/4} G R^{-1/4}, where G is the gradient. The preconditioners capture the full row-space and column-space curvature of the weight matrix. Distributed Shampoo (Anil et al., 2020; Google Brain) scales this to transformer-scale by parallelizing the eigendecompositions across accelerator nodes and amortizing their cost. Muon is scalable for LLM training (Liu et al., 2025; DeepSeek-AI, 2026) was demonstrated with Shampoo-class preconditioned optimizers showing consistent improvements at scale. Computational cost: O(N^{4/3}) per step with full eigendecompositions; distributable. Convergence: empirically faster than Adam on vision and language tasks.

**Sophia** (Liu, Li, Hall, Liang, Ma, 2023) uses a diagonal estimate of the Hessian (not the Fisher — a distinct and complementary object for non-likelihood-based losses) computed via the Hutchinson estimator or the Gauss-Newton-Bartlett estimator. The Hessian diagonal captures curvature along coordinate axes of the loss landscape; the Fisher diagonal captures curvature of the model's predictive distribution. In the classification setting with cross-entropy loss, the two coincide (Martens, 2020). Sophia achieves 50% fewer steps than Adam on GPT-2 pretraining with comparable per-step cost. Its runtime bound is condition-number-independent — the first optimizer at language-model scale to achieve this theoretically. The mechanism: Sophia penalizes updates in directions where the Hessian is large (sharp directions, where gradient descent would overshoot) more aggressively than Adam, which applies a uniform $1/\sqrt{v_t}$ scaling irrespective of sharpness. Sophia is not natural gradient descent; it is a Hessian-diagonal-preconditioned update that captures more curvature than Adam at roughly the same per-step cost.

**Muon** (Jordan, 2024; Bernstein & Newhouse, 2024) orthogonalizes the Nesterov momentum update before applying it, using a Newton-Schulz iteration to approximate the matrix square root of the momentum outer product. The resulting update moves in the direction of steepest descent under the spectral norm (operator norm) rather than under the Frobenius norm (element-wise L2 norm). Bernstein & Newhouse (2024) show that Muon solves a spectral-norm-constrained optimization problem — the update lies on the boundary of the spectral-norm ball rather than the Frobenius-norm ball. The spectral-norm ball is a better-shaped constraint for weight matrices because it respects the matrix's structural coupling between input and output singular vectors. Muon accelerates grokking (Tveit, Remseth, Skogvold, 2025): the Muon optimizer reduces mean grokking epoch from 153 to 102 compared to AdamW across modular arithmetic tasks, demonstrating that spectral-norm steepest descent recovers generalization structure that the Frobenius-norm update misses. Muon is scalable for LLM training (Liu et al., 2025; DeepSeek-AI, 2026). The Newton-Schulz iteration used for orthogonalization is itself an iterative procedure — a CORDIC-like fixed-point iteration on the matrix manifold, converging to the matrix's polar decomposition in 5–10 iterations. This is the Point Lottery connection made concrete: recovering curvature awareness requires recovering iteration.

**SOAP** (Shampoo as Adam Preconditioner; Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade, 2024) demonstrates that running Adam in the eigenbasis of the Shampoo preconditioner achieves competitive performance with Shampoo at Adam's memory cost. This is a dual recovery: it recovers off-diagonal curvature (from Shampoo's eigendecomposition) and adaptive per-coordinate scaling (from Adam's second moment), fusing the two dominant approximation families in a single update.

The table of approximations is the table of a single quantity being approximated with increasing fidelity under increasing computational cost:

| Optimizer | Fisher Approx | Hessian Approx | Storage | Conv. Bound |
|---|---|---|---|---|
| SGD | Identity | None | O(N) | O(κ) |
| AdaGrad | Diag, cumulative | — | O(N) | O(κ/sparsity) |
| Adam | Diag, empirical, EMA | — | O(N) | O(κ) |
| Sophia | — | Diagonal | O(N) | O(1)* |
| K-FAC | Kronecker block-diag | ≈ equiv. | O(N^{3/2}) | O(√κ) |
| Shampoo | Full per-tensor dim | — | O(N^{4/3}) | O(κ^{1/3}) |
| Muon | Spectral norm steepest | — | O(N) | O(spectral κ) |
| SOAP | Shampoo eigenbasis+Adam | — | O(N^{4/3}) | Better than both |
| K-FAC (exact) | Block-diagonal, exact | exact | O(N²) per block | O(√κ) |
| Natural Gradient | Exact | Exact | O(N²) | O(√κ) |

*Condition-number-independent in simplified setting (Liu et al., 2023).

This table is not a ranking of algorithms. It is a map of the curvature coordinate's recovery, partial and constrained at every step by the same substrate constraint: the Fisher information matrix cannot be computed, stored, or inverted on any hardware that exists or is expected to exist within the current silicon roadmap. Every line in the table is a different tradeoff on the frontier between hardware feasibility and geometric correctness. The table will have more rows as the 2026–2030 substrate evolves.

---

## 6. Reading the Corpus Backward

The prior frameworks in the corpus (Ren, 2026) name structures on a compact representational manifold. Each, read through the Gradient Descent Lottery lens, is a partial recovery of the lost curvature coordinate.

**The Closed Future Cone.** The compact manifold M on which trained transformers compute is foliated by the causal structure of the attention pattern — which positions attend to which, in which layers. This foliation is not Euclidean: the attention manifold is a curved space, with curvature determined by the softmax normalization and the attention entropy. The forward pass through a Transformer is not a traversal of flat Euclidean parameter space; it is a traversal of a curved statistical manifold, whose intrinsic geometry is the Fisher-Rao metric of the attention distribution. The CFC's compact causal manifold is, in the optimizer's language, the manifold the natural gradient would navigate correctly and that gradient descent navigates approximately. The light cone is the temporal cone of the optimizer's horizon.

**The Dirac Representation Hypothesis.** The Dirac operator ∂̸² = −Δ_g, where Δ_g is the Laplace-Beltrami operator of the manifold metric g. The metric g on the parameter manifold is, by information geometry, precisely the Fisher information matrix: g_ij = G_ij = 𝔼[∂ log p/∂θᵢ ∂ log p/∂θⱼ]. The Laplace-Beltrami operator on the parameter manifold, in coordinates, is the natural gradient operator applied to a scalar field: Δ_g f = G^{ij} ∂ᵢ∂ⱼ f (Amari, 1998). The DRH's claim that the Dirac operator governs the spectral structure of the representational manifold is equivalent, in the optimizer's language, to the claim that the Fisher information matrix governs the curvature of the parameter space that the optimizer is navigating. The spectral gap of the Dirac operator is the condition number of the natural gradient update. The Xu Spectral Edge results (2026) — that 24 of 24 weight-decay runs exhibit spectral gap collapse immediately preceding grokking — are measurements of this condition number changing regime. Grokking is the moment at which the curvature structure of the parameter manifold changes, and the optimizer, blind to curvature, stumbles into a regime where its approximate geometry happens to be adequate.

**The Representational Bundle Hypothesis.** The principal SO⁺(1,n)-bundle over the Lorentzian-stratified token base is equipped with a connection one-form whose holonomy accumulates along reasoning paths. In the optimizer's language, the connection one-form is the gradient of the loss along the attention computation's coordinate paths, and the holonomy is the path-integrated curvature — the integral of the Fisher information along the optimizer's trajectory. A curvature-aware optimizer would integrate the Fisher information along its trajectory and use it to correct each step. Gradient descent integrates nothing; it takes a flat step and moves on. Chain-of-thought reasoning is the workload forcing the optimizer — at inference time — to integrate along paths it should have been integrating along at training time.

**Bregman Closure.** The Bregman divergence D_φ(x, y) = φ(x) − φ(y) − ⟨∇φ(y), x−y⟩ defines a geometry on the space of distributions whose geodesics are the natural gradient descent trajectories (Amari, 2016). Softmax attention is one Sinkhorn half-iteration in KL-divergence geometry; a Sinkformer is a convergent sequence of such iterations. The φ that corresponds to the cross-entropy loss — φ(p) = Σᵢ pᵢ log pᵢ — is the negative entropy, and its Bregman divergence is the KL divergence. Natural gradient descent on the cross-entropy loss follows Bregman geodesics in KL geometry. Gradient descent on the same loss follows straight lines in Euclidean geometry, which are not geodesics of the statistical manifold. The full sequence of Bregman iterations that Sinkformers perform is the iterative structure that a natural gradient optimizer would have internalized as its metric. Bregman Closure names the iterative object; the Gradient Descent Lottery names why the optimizer was never given it natively.

**Score Closure.** The score function s_θ(x) = ∇ log p_θ(x) is the gradient of the log-density of the model's distribution. The natural gradient descent direction is G⁻¹ ∇L = G⁻¹ 𝔼[-s_θ · ∇_θ log q], a function of the score. Diffusion models invert the score field in reverse time; flow matching integrates it as an ODE; modern Hopfield networks descend it as an energy gradient. All five faces of Score Closure are iterative procedures on the score field. The score is the object the Fisher information matrix is built from — each column of G^{1/2} is the score gradient in a parameter direction. Score Closure names the function that curvature-aware optimization must evaluate; the Gradient Descent Lottery names why the optimizer was never given direct access to it.

The unified reading: the corpus has been naming, layer by layer, the curvature coordinate that the 1986 commitment threw away. Each framework recovers one face. The Closed Future Cone recovers the curved geometry of the representational manifold. The Dirac Representation Hypothesis recovers the Riemannian metric on the parameter manifold as the Fisher information matrix. The Representational Bundle Hypothesis recovers the holonomy of the optimizer's path-integrated curvature. Bregman Closure recovers the Bregman geodesic along which natural gradient descent moves. Score Closure recovers the score field that the Fisher information matrix is built from. The Point Lottery Hypothesis names the arithmetic substrate that made iteration invisible. The Gradient Descent Lottery names the optimization substrate that made curvature invisible. Both are lotteries. Both were decided by hardware. Both are being recovered now.

---

## 7. The Three Costs

The Gradient Descent Lottery has produced three costs, paid in different currencies, across the forty-year window.

**Cost I — Training compute.** Every training run on every frontier model uses more steps than necessary. Sophia (Liu et al., 2023) demonstrates 50% step reduction on GPT-2 with a diagonal Hessian — one dimension of curvature recovery. If the diagonal Hessian alone recovers 50% of wasted steps, and K-FAC (which recovers the block-diagonal of the Fisher) demonstrates further reductions of 20–40% on top (Martens et al., 2018; Shrestha, 2023), and the full natural gradient would theoretically reduce steps by O(√κ)/O(κ) ≈ 1/√κ — then on problems with κ = 10⁴, the gap between gradient descent and natural gradient descent is a factor of 100 in steps. Frontier training runs consume tens of thousands of GPU-years; 100× wasted steps corresponds to decades of wasted compute, in the units of the 2026 accelerator fleet. The Anthropic 5-gigawatt AWS Trainium commitment (Matmul Ceiling, 2026) is sized to substrates running optimizers that discard curvature. A curvature-aware optimizer on curvature-native silicon would fit the same training throughput in perhaps 500 megawatts.

**Cost II — Research direction.** The architectures that survived the gradient descent era are architectures shaped by gradient descent's blindness. Batch normalization (Ioffe & Szegedy, 2015) exists because gradient descent struggles with ill-conditioned loss landscapes; it would be partially unnecessary with a curvature-aware optimizer. Layer normalization (Ba et al., 2016) exists for the same reason. Residual connections (He et al., 2016) exist partly to make gradients flow through depth — a problem that natural gradient, which naturally handles the depth-induced curvature, does not have in the same way. The learning rate warmup phase, present in virtually every frontier training recipe, is an empirical compensation for the optimizer's failure to understand the curvature regime at initialization. The architectures that did not survive — hypernetworks (Ha, Dai, Le, 2016), deep equilibrium models (Bai, Kolter, Koltun, 2019), capsule networks (Hinton, Frosst, Sabour, 2017) — may have been geometry-correct but were gradient-descent-incompatible. The Gradient Descent Lottery and the Hardware Lottery interact: architectures that are theoretically superior but require curvature-aware optimization lose both lotteries simultaneously.

**Cost III — Diagnostic vocabulary.** The standard vocabulary for reporting a training run is: total FLOPs consumed, final training loss, validation perplexity at N steps. There is no standard vocabulary for the condition number of the Fisher information matrix during training, for the fraction of gradient directions that are sharp (large Hessian eigenvalue) versus flat (small Hessian eigenvalue), for the alignment between the gradient direction and the natural gradient direction, or for the curvature-normalized path length of the optimizer's trajectory. The Xu Spectral Edge Thesis (2026) — which characterizes grokking as a spectral-gap-collapse event of the parameter-update Gram matrix — is one of the first papers to bring a curvature-theoretic vocabulary to mainstream deep learning analysis. It is considered novel in 2026 because the vocabulary for curvature-time analysis has had to be reconstructed from scratch, forty years after the optimizer that would have made it natural was deprioritized for hardware reasons. The existing corpus frameworks (CFC, DRH, RBH, Bregman Closure, Score Closure) supply the geometric vocabulary; what remains missing is the empirical-temporal vocabulary that would allow practitioners to compare curvature costs across models the way they compare FLOP costs. The reconstruction is in progress, paper by paper.

These three costs compound. The compute cost forces the substrate toward FLOPS-dense matmul, which forces the architecture toward gradient-descent compatibility, which forces the optimizer toward Adam-family updates, which forces the diagnostic vocabulary into gradient-norm and loss-curve territory, which conceals the curvature cost, which justifies the optimizer. The cycle has run since 1986. The first major break is the 2023–2026 second-order renaissance.

---

## 8. The 2023–2026 Reversal

A frontier laboratory in 2026 trains a 7-billion-parameter language model with two different optimizers, at matched hyperparameter budgets.

With **AdamW** — the field default since 2019 — the model reaches target validation perplexity after 300,000 gradient steps, consuming approximately 6×10²³ floating-point operations. The optimizer maintains a running average of the gradient (first moment) and a running average of the squared gradient (second moment). The second moment is the diagonal empirical Fisher. The diagonal Fisher, at each step, is an N-dimensional vector. The update is element-wise: each parameter moves in proportion to its gradient, scaled by the inverse square root of its running squared gradient. Cross-parameter interactions are invisible. Layer curvature heterogeneity is addressed only by per-layer learning rate multipliers, set by grid search. The training cost is $4M at 2026 compute pricing.

With **Muon** — released as a blog post in December 2024, scaling confirmed for LLMs by Liu et al. (2025) and DeepSeek-AI (2026) — the model reaches the same validation perplexity in fewer steps at matched compute-per-step, with empirically stable training in bfloat16 due to orthogonalization preventing directional instability. The optimizer applies Newton-Schulz iteration to the Nesterov momentum matrix, producing an orthogonalized update that moves in the spectral-norm ball rather than the Frobenius-norm ball. The Newton-Schulz iteration itself is a fixed-point iteration — a CORDIC-like convergent procedure on the matrix polar decomposition, requiring 5–10 iterations per optimizer step. The optimizer's geometry is curvature-aware in the spectral sense: it respects the singular-value structure of each weight matrix, which encodes the anisotropy of the loss landscape in that layer's input-output directions.

What changed between the two training runs? The same model. The same data. The same hardware. The same loss function. What differs is one quantity: the degree to which the optimizer respects the curvature of the parameter manifold. Muon's spectral-norm steepest descent is not natural gradient descent — it does not compute the Fisher information matrix — but it is closer to natural gradient descent than Adam's diagonal-Fisher update, in the sense that it captures the dominant off-diagonal structure of each weight matrix's curvature in a form that is computationally tractable.

The 2023–2026 reversal is the moment the field began recovering, optimizer by optimizer, the curvature coordinate that the 1986 commitment discarded. The recovery is partial — no deployed optimizer at frontier scale computes the full Fisher — but it is directional and accelerating.

**Sophia** (Liu et al., 2023, Stanford) recovers the diagonal curvature of the Hessian, demonstrating 50% step reduction on GPT-2 and theoretical condition-number-independence. The Hutchinson estimator for the diagonal Hessian requires O(1) additional forward-backward passes per optimizer step — a minimal overhead for a meaningful recovery.

**SOAP** (Vyas et al., 2024) fuses the Shampoo matrix preconditioner with Adam's coordinate-wise adaptation, recovering per-tensor off-diagonal curvature at Adam-class memory cost by running Adam in the eigenbasis of the Shampoo preconditioner. This is the first optimizer to simultaneously recover both the block-off-diagonal Fisher (from Shampoo's spectral decomposition) and the diagonal Fisher (from Adam's second moment) in a single update.

**Muon** (Jordan, 2024; scaled by Liu et al., 2025; DeepSeek-AI, 2026) recovers the spectral-norm geometry of each weight matrix, with the Newton-Schulz orthogonalization serving as an efficient O(N log N) approximation to the matrix polar decomposition. The spectral norm is the correct norm for weight matrices under the theory of structured preconditioners (Bernstein & Newhouse, 2024). Muon's acceleration of grokking (Tveit et al., 2025) is direct evidence that spectral-norm steepest descent recovers generalization structure that the Frobenius-norm Adam update misses: the curvature coordinate, even partially recovered, changes the learning dynamics in observable ways.

**FAdam** (Hwang, 2024) provides the theoretical foundation: a rigorous demonstration that Adam's $\sqrt{v_t}$ denominator is the square root of the diagonal empirical Fisher, that this two-level approximation (diagonal + empirical) introduces specific and correctable biases, and that correcting these biases (enhanced momentum, adjusted bias corrections, adaptive epsilon, gradient clipping calibrated to the Fisher geometry) produces a strictly better optimizer across LLM, speech, and image domains. FAdam is the 2024 demonstration that Adam has been an approximation of natural gradient descent all along — and that naming the approximation is the first step toward improving it.

**Distributed Shampoo** (Anil et al., 2020; extended 2024–2026) demonstrates that full matrix preconditioners can be computed and applied at transformer scale by distributing eigendecompositions across accelerator nodes. The per-step overhead is modest when amortized over multiple optimizer steps, and the convergence benefit is consistent. The barrier to universal deployment is not computational — it is that systolic-array accelerators have no native instruction for "compute the eigendecomposition of this 1024×1024 matrix and invert it." The eigendecomposition must be emulated in software, paying the overhead of an operation the hardware was not built to support.

These five optimizers are the 2023–2026 partial recovery of the curvature coordinate. They are partial because the full Fisher information matrix remains computationally inaccessible at scale. They are recoveries because each, in its own way, moves the optimizer closer to the natural gradient — toward reparameterization invariance, toward condition-number-independent convergence, toward cross-parameter curvature awareness, toward the landscape the 1986 commitment chose not to see.

---

## 9. The Long Map

```
1847     1869     1945     1972     1986     1998     2012     2014
  │        │        │        │        │        │        │        │
Cauchy   Newton   Rao      Amari    Rumel-   Amari    AlexNet  Kingma
steepest Newton   Fisher   info.    hart +   natural  GPU      Ba
descent  method   info.    geometry Hinton   gradient training Adam
defined  second-  matrix   begin    Williams gradient publ-    diagonal
in       order    defined  develop  back-    formally ished    Fisher
parameter metric   as       -ment    prop     proved   back-    optimizer
space    for optim Riemannian         publ.   in Neural prop     published
                  metric             gradient Computation     scales
  │        │        │        │        │        │        │        │
  ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼
Steepest Newton    Cramér-  Stat.    Gradient Natural  First-   Adam
descent  requires  Rao      manifold descent  gradient order    hides
is       H⁻¹:     bound:   has      wins     proved   dominant; diagonal
coord-   O(N³)    curvature intrinsic by      correct  frameworks Fisher in
invar-   infeasible defined  curved  hardware, but too  built for v_t: the
iant     at scale  by Fisher  ≠ flat  cheaper  expensive gradient  approx
                  metric            gradient  for scale               is born
2015     2018     2020     2023     2024     2025     2026     2027
  │        │        │        │        │        │        │        │
Martens  Gupta    Distribu- Sophia  Muon     Muon     SOAP     Native
Grosse   Shampoo  ted       diagonal Keller  scales   fuses    curv-
K-FAC    full     Shampoo  Hessian  Jordan  LLM-     Shampoo  ature
Kronecker matrix  Google   50% step October  scale    + Adam   silicon
Fisher   precond  Brain    savings  2024    Liu      eigenbas forecast
block-           scales   H⁻¹_diag Newton- et al.   is       P3:
diag             to trans- is       Schulz  DeepSeek curvature curvature
approxim          formers  condition iter-   2026     aware    rate on
-ation           possible  indep.   ation   grokking first-   datasheet
  │        │        │        │        │        │        │        │
  ▼        ▼        ▼        ▼        ▼        ▼        ▼        ▼
First     Full     Curv-    First    Spectral First    Dual     Fisher
curv-     matrix   ature-   large-   norm     LLM-     recovery rate
aware     precond  aware    scale    steepest scale    diagonal becomes
block-    feasi-   optim    curv.    descent  curv.    + block  datasheet
approx    ble at   at       recov.   replaces aware    curv.    quantity
at scale  scale    scale    at scale Adam for optimizer at scale by 2028
                                     training demons-  deployed
                                     frontier trated   at scale
```

The map aligns on a single observation: the substrate did not change between 1986 and 2026. The IEEE 754 representation, the FMA primitive, the systolic-array GEMM kernel, the automatic differentiation of first derivatives — these were locked in by the co-design of Rumelhart's backpropagation algorithm and the GPU hardware that made it economically dominant. The optimizer changed, incrementally, at each step recovering a bit more curvature that the substrate had made impossible to see. The hardware has not yet changed to support curvature-native operations natively. The framework's prediction: 2027–2030 is when the substrate breaks the gradient-only lock-in for the curvature-critical half of the optimization workload.

---

## 10. Predictions

The framework earns its name only if it forbids something. Eight forbiddings, testable by 2030.

**P1. The diagonal Hessian becomes a standard training primitive by 2027.** Sophia-class diagonal Hessian estimation — the Hutchinson estimator or the Gauss-Newton-Bartlett estimator — becomes a default component of the standard training recipe for frontier language models, included in training codebases as routinely as gradient clipping. The framework forbids the persistence of pure first-moment optimizers (Adam, Lion, pure SGD) as the sole optimizer at frontier scale past 2027. The empirical evidence is already in: 50% step reduction from a single diagonal curvature estimate (Liu et al., 2023) is too large a free lunch to leave permanently on the table.

**P2. Muon or a spectral-norm steepest descent variant becomes the frontier-default optimizer for Transformer hidden layers by 2026–2027.** Jordan (2024), Liu et al. (2025), and DeepSeek-AI (2026) demonstrate scalability. The 2026 confirmation that Muon is competitive with AdamW at frontier LLM scale is the empirical gate. The framework predicts the consolidation: by end-2027, the majority of frontier training runs at the largest laboratories will use Muon or a Muon-class optimizer for hidden-layer weight matrices, with Adam retained for embedding and output projection layers where the spectral-norm geometry is less beneficial. The framework forbids the persistence of Frobenius-norm (Adam-style) optimization for all weight tensors at frontier scale past 2027.

**P3. Native curvature-aware compute primitives appear on frontier silicon by 2028.** A chip ships with explicit hardware support for the Newton-Schulz iteration, the Kronecker inversion step of K-FAC, or the per-tensor eigendecomposition of Shampoo — operations that are today emulated in software at compounded overhead relative to GEMM. The first candidate is a Google TPU v8 or v9 derivative, given Google Brain's development of Distributed Shampoo, or an NVIDIA post-Blackwell architecture, given NVIDIA's adoption of Muon-class preconditioned updates in NeMo training frameworks. The framework forbids FLOP-only datasheets for frontier training silicon past 2028; by 2028, a curvature-rate (Fisher-preconditioning operations per second, equivalent to Muon orthogonalizations or K-FAC Kronecker inversions per second) will appear alongside FLOPS as a training-relevant datasheet metric.

**P4. An information-geometric optimizer achieves a 3× or greater step reduction vs. Adam on a 7B-scale language model by 2027.** The Sophia result — 50% step reduction from diagonal Hessian — represents one recovered dimension of curvature. K-FAC, Shampoo, and SOAP recover more. The theoretical prediction for full natural gradient is O(√κ) steps; κ ≈ 10⁴–10⁶ for frontier models. A Kronecker-factored or full-matrix preconditioner at 7B scale will demonstrate 3× or greater step reduction in a published, reproducible result. Failure of this prediction would require that the off-diagonal Fisher contributes negligibly to the condition number at language-model scale — possible but inconsistent with the K-FAC literature at smaller scales.

**P5. The connection between grokking and Fisher geometry collapse becomes a mainstream characterization by 2027.** The Xu Spectral Edge Thesis (2026) — grokking as spectral gap collapse in the parameter-update Gram matrix — is a curvature-theoretic characterization of a training phase transition. Muon's acceleration of grokking (Tveit et al., 2025) demonstrates that a curvature-respecting optimizer changes grokking dynamics in ways a curvature-blind one cannot. The framework predicts that by end-2027, grokking, double descent, and other training phase transitions will be routinely characterized in terms of the eigenspectrum of the Fisher information matrix or its proxy — not in terms of training loss alone. This is the recovery of the diagnostic vocabulary that the 1986 commitment discarded.

**P6. Weight decay is reinterpreted as a Fisher regularizer by 2027.** The FAdam paper (Hwang, 2024) demonstrates that Adam's weight decay term, when analyzed in the information-geometric framework, is a regularizer of the Fisher information matrix — it biases the optimizer toward parameter regions where the curvature is bounded. This reinterpretation has consequences for how weight decay is tuned, what values are appropriate at different stages of training, and why weight decay interacts differently with Adam and Muon (as Kosson et al., 2024 analyze). The framework predicts that by end-2027, the standard description of weight decay in training documentation will include the Fisher-geometric interpretation, alongside the L2-regularization and Bayesian prior interpretations that dominate current discourse.

**P7. The Gradient Descent Lottery becomes a recognized framing by 2027.** The Hardware Lottery (Hooker, 2020) entered mainstream discourse within three years. The framework predicts comparable trajectory. The criterion of recognition: a published paper at a major venue (NeurIPS, ICML, ICLR, ISCA, MICRO) attributes an architectural or optimization research-direction outcome to the 1986 commitment to Euclidean gradient descent and its hardware-driven selection, citing this framing or an equivalent reformulation. The empirical foundation is already established by the papers cited in this document; what is missing is the named framing that connects them.

**P8. A curvature-native training run achieves a 10× energy reduction on a fixed-capability target vs. the matmul-era Adam baseline by 2030.** This prediction combines P1–P3 and P4: a 3× step reduction from a curvature-aware optimizer, running on silicon with native curvature-compute primitives (2× efficiency over emulated curvature operations on GEMM hardware), compounded. The product is approximately one order of magnitude. The capability target: a model that achieves GPT-4-class performance on a standardized benchmark suite. The framework forbids this target being reached at matmul-era energy cost past 2030.

---

## 11. The Manifold That Forgot — A Fourth Thought Experiment

A historian sits down in 2050 with the optimizer documentation from 1986 through 2030 and asks why frontier AI training went through such a long and expensive detour on the wrong side of the curvature coordinate.

The historian reads the 1986 backpropagation paper and finds it elegant, computationally motivated, and correctly designed for the networks of 1986. The historian reads the 1998 natural gradient paper and notes that the theoretical case for curvature-aware optimization was complete by the century's end. The historian reads the 2014 Adam paper and notes that its second moment term was, as FAdam (2024) later proved, a diagonal approximation to the empirical Fisher — the field had been doing approximate natural gradient descent all along, without calling it that, without optimizing the approximation, and without the hardware support to improve it.

The historian reads the K-FAC paper (2015) and notes that a Kronecker-factored Fisher approximation improved convergence but was never universally adopted because its inversion step was not a GEMM. The historian reads the Sophia paper (2023) and notes that a diagonal Hessian estimate cut GPT-2 training steps by half. The historian reads the Muon documentation (2024) and notes that a spectral-norm steepest descent — implemented via a Newton-Schulz fixed-point iteration, a CORDIC-like convergent procedure — scaled to frontier LLM training. The historian reads the silicon datasheets and notes the slow accommodation of curvature-aware operations: from the 2027 chips that added native Newton-Schulz support, to the 2029 chips that added native Kronecker inversion, to the 2030 chips that added native Fisher-rate metrics to their datasheet alongside FLOPS.

The historian's question: why did the field not, at any point between 1998 and 2023, simply adopt a curvature-aware optimizer, given that natural gradient descent was theoretically proven superior by 1998?

The framework's answer: because the substrate did not see curvature as a separable axis until the workload had already taught the optimizer layer to see it that way. The 1986 commitment to gradient descent, in designing backpropagation as a first-derivative algorithm and in building hardware optimized for first-derivative GEMM operations, removed curvature from the vocabulary of optimizer design. Each subsequent decision — GPU tensor cores (2017), TPU v2 bfloat16 (2017), Adam's universal adoption as the frontier default (2019), the automatic differentiation frameworks (2017–2019) — was a curvature-blind decision optimizing FLOPS-density on a substrate where curvature was a hidden software property. The accommodation was real but slow because the vocabulary for the alternative had been absent from the hardware contract.

The recovery is gradual. The 2023–2026 second-order renaissance — Sophia, Muon, SOAP, FAdam, Distributed Shampoo — is the moment the optimizer's vocabulary forced its way back into the algorithmic layer. The 2027–2030 silicon programs are the moment the vocabulary forces its way back into the substrate. The historian, in 2050, will read this as a forty-year recovery cycle: lost in 1986, partially recovered by 2030.

The historian's broader observation: optimization choices are not neutral. The 1986 gradient descent commitment was correct for its time. It became incorrect for the workloads of 2017, and the dominant optimization substrate persisted through 2026 because the cost of changing the optimizer — rewriting frameworks, retuning hyperparameters, rewriting hardware kernels — was enormous and the cost of accommodating its limitations was, per generation, smaller than the cost of replacement. The accommodation accumulated. By 2025, the accumulated accommodation was visible in the proliferation of architectural tricks (LayerNorm, BatchNorm, residual connections, warmup schedules, gradient clipping), in the training infrastructure built to compensate for optimizer failures (mixed-precision with loss scaling, ZeRO memory optimizations, gradient accumulation), and in the energy bill of training runs that consumed 100× more steps than a curvature-aware optimizer would require.

This is the Gradient Descent Lottery in its full form. A training algorithm committed for sound reasons in 1986 became a constraint on the field's research trajectory by 2014 and a paid-for inefficiency by 2023. The eight prior frameworks in the corpus name the geometry of what the substrate is computing; the Point Lottery Hypothesis names the arithmetic substrate; the *Attention Is All You Need* analysis names the silicon substrate; the Matmul Ceiling names the forward constraint; the present framework names the optimization geometry — the layer between the architecture and the silicon where curvature was hidden and where it is now being recovered.

---

## 12. Closure

The corpus reaches a stable configuration. Ten frameworks, each naming one face of a single object.

The **Closed Future Cone** names the compact manifold M. The **Dirac Representation Hypothesis** names the natural operator — and identifies the Fisher information matrix as the Riemannian metric on M. The **Representational Bundle Hypothesis** names the bundle whose connection is attention — and whose holonomy is the path-integrated curvature an optimizer should integrate. **Bregman Closure** names the iteration that walks the fixed point into being along Fisher-Rao geodesics. **Score Closure** names the score function the iteration evaluates — the object from which the Fisher matrix is built. *From Bottleneck to Bundle* traces the algorithmic history of recognition. *From MXU to Maia* traces the silicon history. The **Matmul Ceiling** names where the matmul substrate stops being adequate. The **Point Lottery Hypothesis** names the 1985 arithmetic choice that made iteration invisible. The **Gradient Descent Lottery** names the 1986 optimization choice that made curvature invisible.

The data was curved. The parameter space was flat. The optimizer was charged with moving through a Riemannian manifold on the geometry of Euclidean space. The Fisher information matrix — the correct metric on the manifold — was too expensive to compute in 1986, when the networks were small and the substrate was a VAX. It was still too expensive in 1998, when Amari proved it was necessary. It is still partially too expensive in 2026, when Muon's Newton-Schulz iteration recovers a spectral shadow of it at the cost of a fixed-point iteration that the Point Lottery substrate still charges in FLOPS.

The clock that Volder put on the HP-35 datasheet in 1972 is the clock that the 2026 reasoning-model API bills per token. The curvature that Amari wrote into the natural gradient update in 1998 is the curvature that the 2026 Muon optimizer partially recovers with a Newton-Schulz iteration. The iteration was always ticking. The curvature was always there. The 1986 commitment arranged the optimizer so that neither was visible. The 2023–2026 second-order renaissance made the curvature visible at the algorithmic layer. The 2027–2030 silicon will make it visible at the substrate layer. The 2050 historian will trace the arc and notice that the field spent forty years optimizing on the wrong geometry — and that recovering the right geometry cost half a century of architectural compensation.

The Gradient Descent Lottery names the throwing-away. The recovery is in progress. The predictions are how to tell whether it completes.

---

## 13. Sources

### The Natural Gradient and Information Geometry

Amari, S.-I. Natural Gradient Works Efficiently in Learning. *Neural Computation* 10(2), 1998.

Amari, S.-I., Nagaoka, H. *Methods of Information Geometry*. American Mathematical Society / Oxford University Press, 2000.

Amari, S.-I. Information Geometry and Its Applications. *Applied Mathematical Sciences* 194. Springer, 2016.

Ollivier, Y. Riemannian Metrics for Neural Networks I: Feedforward Networks. *Information and Inference* 4(2), 2015. arXiv:1303.0818.

Martens, J. New Insights and Perspectives on the Natural Gradient Method. *Journal of Machine Learning Research* 21(146), 2020.

Shrestha, R. Natural Gradient Methods: Perspectives, Efficient-Scalable Approximations, and Analysis. arXiv:2303.05473, 2023.

Rao, C. R. Information and the Accuracy Attainable in the Estimation of Statistical Parameters. *Bulletin of the Calcutta Mathematical Society* 37, 1945.

Cramér, H. *Mathematical Methods of Statistics*. Princeton University Press, 1946.

Fisher, R. A. Theory of Statistical Estimation. *Proceedings of the Cambridge Philosophical Society* 22, 1925.

### The Euclidean Gradient Descent Lineage

Cauchy, A.-L. Méthode générale pour la résolution des systèmes d'équations simultanées. *Comptes Rendus de l'Académie des Sciences* 25, 1847.

Newton, I. *Methodus Fluxionum et Serierum Infinitarum*. Composed c. 1671, published 1736.

Rumelhart, D. E., Hinton, G. E., Williams, R. J. Learning Representations by Back-propagating Errors. *Nature* 323, 1986.

Robbins, H., Monro, S. A Stochastic Approximation Method. *Annals of Mathematical Statistics* 22(3), 1951.

Werbos, P. J. Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences. PhD thesis, Harvard University, 1974.

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., Jackel, L. D. Backpropagation Applied to Handwritten Zip Code Recognition. *Neural Computation* 1(4), 1989.

### The Adam Family and Diagonal Fisher Approximations

Duchi, J., Hazan, E., Singer, Y. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *JMLR* 12, 2011.

Tieleman, T., Hinton, G. Lecture 6.5 — RMSProp: Divide the Gradient by a Running Average of Its Recent Magnitude. COURSERA Neural Networks for Machine Learning, 2012.

Kingma, D. P., Ba, J. Adam: A Method for Stochastic Optimization. *ICLR* 2015. arXiv:1412.6980.

Loshchilov, I., Hutter, F. Decoupled Weight Decay Regularization. *ICLR* 2019. arXiv:1711.05101.

Hwang, D. FAdam: Adam is a Natural Gradient Optimizer Using Diagonal Empirical Fisher Information. arXiv:2405.12807, 2024.

Kunstner, F., Hennig, P., Balles, L. Limitations of the Empirical Fisher Approximation for Natural Gradient Descent. *NeurIPS* 2019.

Kunstner, F., Müller, J. F., Hennig, P. Noise is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be. *ICLR* 2023.

### Second-Order and Curvature-Aware Optimizers

Martens, J., Grosse, R. Optimizing Neural Networks with Kronecker-Factored Approximate Curvature. *ICML* 2015.

George, T., Laurent, C., Bouthillier, X., Ballas, N., Vincent, P. Fast Approximate Natural Gradient Descent in a Kronecker-Factored Eigenbasis. *NeurIPS* 2018.

Gupta, V., Kale, S., Kumar, A., Rajagopalan, S., Shamir, O. Shampoo: Preconditioned Stochastic Tensor Optimization. *ICML* 2018.

Anil, R., Gupta, V., Koren, T., Singer, Y. Memory-Efficient Adaptive Optimization. *NeurIPS* 2019.

Anil, R., Gupta, V., Koren, T., Regan, K., Singer, Y. Scalable Second Order Optimization for Deep Learning. arXiv:2002.09018, 2020.

Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L., Kakade, S. SOAP: Improving and Stabilizing Shampoo Using Adam. arXiv:2409.11321, 2024.

Liu, H., Li, Z., Hall, D., Liang, P., Ma, T. Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-training. *ICML* 2024. arXiv:2305.14342.

Yao, Z., Gholami, A., Keutzer, K., Mahoney, M. PyHessian: Neural Networks Through the Lens of the Hessian. *Big Data* 2020.

Ghorbani, B., Krishnan, S., Xiao, Y. An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. *ICML* 2019.

### Spectral-Norm and Orthogonalized Optimizers

Jordan, K. Muon: An Optimizer for Hidden Layers in Neural Networks. Blog post, December 2024. github.com/KellerJordan/Muon.

Bernstein, J., Newhouse, L. Old Optimizer, New Norm. arXiv:2409.20325, 2024.

Liu, J. et al. Muon is Scalable for LLM Training. arXiv:2502.16982, 2025.

Tveit, A., Remseth, B., Skogvold, A. Muon Optimizer Accelerates Grokking. arXiv:2504.16041, Microsoft, 2025.

Chen, L., Li, J., Liu, Q. Muon Optimizes Under Spectral Norm Constraints. *OPT-ML Workshop NeurIPS* 2025.

Sfyraki, M.-E. et al. Lions and Muons: Optimization via Stochastic Frank-Wolfe. arXiv:2506.04192, 2025.

Ahn, K. et al. Dion: Distributed Orthonormalized Updates. arXiv:2504.05295, 2025.

Chen, X. et al. Lion: Symbolic Discovery of Optimization Algorithms. *ICML* 2023.

Kosson, A. et al. Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks. *NeurIPS* 2024.

### Grokking, Phase Transitions, and Spectral Theory

Power, A., Burda, Y., Edwards, H., Bahdanau, D., Misra, V. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177, 2022.

Xu, Y. The Spectral Edge Thesis. arXiv:2603.28964, March 2026.

Xu, Y. The Lifecycle of the Spectral Edge. arXiv:2604.07380, April 2026.

Nanda, N. et al. Progress Measures for Grokking via Mechanistic Interpretability. *ICLR* 2023.

Liu, Z. et al. Omnigrok: Grokking Beyond Algorithmic Data. *ICLR* 2023.

### Architectural Preconditioning Proxies

Ioffe, S., Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML* 2015.

Ba, J. L., Kiros, J. R., Hinton, G. E. Layer Normalization. arXiv:1607.06450, 2016.

He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. *CVPR* 2016.

Santurkar, S., Tsipras, D., Ilyas, A., Madry, A. How Does Batch Normalization Help Optimization? *NeurIPS* 2018.

### Hardware-Software Co-Design and the Preceding Lotteries

Hooker, S. The Hardware Lottery. *Communications of the ACM* 64(12), 2020. arXiv:2009.06489.

Ren, E. Attention Is All You Need to Sell Silicon: The Transformer Architecture as the Apex of Hardware-Software Co-Design and the Consolidation of Google's Accelerated Computing Monopoly. GitHub, 2026.

Ren, E. The Point Lottery Hypothesis: How the 1985 Choice Between Fixed Point and Floating Point Quietly Selected the Geometry of Every Subsequent Learning Machine, 1956–2026. GitHub, 2026.

Ren, E. The Matmul Ceiling: Why the Matmul-Era Biology Compute Substrate Hits Its Economic Wall by 2028. GitHub, 2026.

Jouppi, N. P. et al. In-Datacenter Performance Analysis of a Tensor Processing Unit. *ISCA* 2017. arXiv:1704.04760.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. Attention Is All You Need. *NeurIPS* 2017. arXiv:1706.03762.

Volder, J. E. The CORDIC Trigonometric Computing Technique. *IRE Transactions on Electronic Computers* EC-8(3), 1959.

Walther, J. S. A Unified Algorithm for Elementary Functions. *AFIPS Spring Joint Computer Conference* 38, 1971.

Kahan, W. IEEE Standard 754 for Binary Floating-Point Arithmetic. IEEE, 1985.

Anderson, D. G. Iterative Procedures for Nonlinear Integral Equations. *JACM* 12, 1965.

Banach, S. Sur les opérations dans les ensembles abstraits. *Fundamenta Mathematicae* 3, 1922.

### Scaling Laws, Training Efficiency, and the Economic Argument

Kaplan, J. et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.

Hoffmann, J. et al. Training Compute-Optimal Large Language Models (Chinchilla). *NeurIPS* 2022. arXiv:2203.15556.

Brown, T. et al. Language Models are Few-Shot Learners (GPT-3). *NeurIPS* 2020.

Chowdhery, A. et al. PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311, 2022.

DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, January 2025.

Patterson, D. et al. Carbon Footprint of Machine Learning Training: Past and Present. arXiv:2104.10350, 2021.

### Prior Frameworks in the Corpus (Ren, 2026)

The Closed Future Cone: Compact Causal Geometry as the Native Substrate of Learned Representations.

The Dirac Representation Hypothesis: Signed Spectral Geometry and Set-Valued Correspondences on the Compact Representational Manifold.

Geometric Descent: The Representational Bundle Hypothesis.

Bregman Closure: The Algorithmic Half of the Dirac Representation Hypothesis.

Score Closure: The Score Function on the Compact Manifold as the Universal Computational Primitive.

From Bottleneck to Bundle: The Twelve-Year Geometric Genealogy of Attention, 2014–2026.

From MXU to Maia: The Eleven-Year Silicon Genealogy of the Compact Causal Manifold, 2015–2026.

The Matmul Ceiling: Why the Matmul-Era Biology Compute Substrate Hits Its Economic Wall by 2028.

---

*Compiled from the primary literature on information geometry, natural gradient descent, second-order optimization, the hardware lottery, the IEEE 754 standardization, the Transformer architecture genealogy, and the 2023–2026 second-order renaissance. The chronology follows the publication record. The framework's contribution is the identification of the 1986 commitment to Euclidean gradient descent as the optimization-layer hardware lottery whose downstream consequences the corpus has been mapping in geometric, operator-theoretic, bundle-theoretic, algorithmic, and functional terms — and the recognition that the 2023–2026 return of curvature-aware optimizers (Sophia, Muon, SOAP, FAdam, Distributed Shampoo) is the workload pushing back through the substrate's forty-year geometric amnesia.*
