---
title: "Modelling Bounded Rationality in Multi-Agent Interactions by Generalized Recursive Reasoning"
path: "/GR2"
tags: ["Paper Note"]
featuredImage: "./cover.jpg"
excerpt: Paper summery of GR2
created: 2019-02-07
updated: 2019-02-07
---

Paper URL: <a href="https://arxiv.org/abs/1901.09216">https://arxiv.org/abs/1901.09216</a> by Ying Wen, Yaodong Yang, Rui Luo, Jun Wang

# Modelling Bounded Rationality in Multi-Agent Interactions by Generalized Recursive Reasoning



_TODO_

- Redo all the proofs -- so that i matched the same patterns.
- Do the algorithms
- Quote from paper



## Formulation

The algorithm (GR2) assume that its opponent are less sophisticated than them, thinking about best response to its opponent. This can be formulated as following
$$
\pi^i_k(a^i_k | s) \propto \int_{a^{-i}_{k-1}} \left\{ \pi^i_k(a^i_k | s, a^{-i}_{k-1}) \int_{a^i_{k-2}} \rho^{-i}_{k-1}(a^{-i}_{k-1} | s, a^i_{k-2}) \pi^i_{k-2}(a^i_{k-2} | s) \ da^i_{k-2} \right\} da^{-i}_{k-1}
$$
So the agent at level $k$ is best responding to its opponent that is best responding to agent that thinking at level $k-2$. We assume that the model for all the level are the same
$$
\pi^i_{\theta^k} = \pi^i_{\theta^{k+2}} \quad \quad \rho^{-i}_{\theta^k} = \rho^{-i}_{\theta^{k+2}}
$$
and the base policy is assumed to be uniformly distributed. Now, the author make an assumption, where: with increasing $k$, level-$k$ agents have an accurate guess about relative propotion of agent's who are doing lower level thinking than them. This helps the calculation since, when $k$ is large, there is no benefit of level-$k+1$ to be calculated, since they are similar.

This assumption can be represented by Poission distribution $f(k) = \frac{e^{-\lambda}\lambda^k}{k!}$ where they are now mixing each level thinking
$$
\pi^{i, \lambda}_k(a^i_k | s, a^{-i}_{0:k-1}) = \frac{e^{-\lambda}}{Z} \left( \frac{\lambda^0}{0!} \hat{\pi}^i_0(a^i_0|s)+ \frac{\lambda^1}{1!}\hat{\pi}^i_1(a^i_1|s) + \cdots + \frac{\lambda^k}{k!}  \hat{\pi}^i_k(a^i_k|s)\right)
$$
where $Z = \sum^k_{i=1} e^{-\lambda} \lambda^n / n!$ which is a normalizing factor.



## Implementation

The parameter $\omega^i$ of joint soft-Q function is trained by minimizing the following Bellman residual loss:
$$
\mathcal{J}_{Q^{i}}(\omega^i) = \mathbb{E}_{(s, a^i, a^{-i}) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q^i_{\omega^i}(s, a^i, a^{-i}) - \hat{Q}^i(s, a^i, a^{-i}) \right)^2\right]
$$
Where the target is $\hat{Q}^i(s, a^i, a^{-i}) = r^i(s, a^i, a^{-i}) + \gamma \mathbb{E}_{s' \sim p_s}[V(s')]$. The value function for level $k$ policy can is
$$
V^{i}(s) = \mathbb{E}_{a^i \sim \pi^i_k}\left[ Q^i(s, a^i) - \log \pi^i_k(a^i | s) \right]
$$
The partial Q-function $Q^{i}(s, a^i)$ is calculated via marginalization the joint Q-function using estimated opponent model $\rho^i_{\phi^{-i}}$
$$
Q^{i}(s, a^i) = \log \int\rho^i_{\phi^{-i}}(a^{-i} | s, a^i) \exp\left( Q^i(s, a^i, a^{-i}) \right) \ da^{-i}
$$

---

For opponent model, the loss is calculated as
$$
\mathcal{J}_{\rho^{-i}} (\phi^{-i}) = D_{KL}\left[ \rho^{-i}_{\phi^{-i}}(a^{-i} | s, a^i) \Bigg\| \exp\left( Q^i_{\omega^i}(s, a^i, a^{-i}) - Q^i_{\omega^i}(s, a^i) \right) \right]
$$
For both of $Q$-function, the authors use 2 approximate functions, and the gradient $\phi^i$ is completed via SVGD.

---

For level-$k$ policy parameter $\theta^i$ is trained by improving toward Q-function $Q^{i}_\omega(s, a^i)$ with re-paramterized trick $a^i = f_{\theta^i}(\varepsilon ; s)$ as minimizing the following loss function
$$
\mathcal{J}_{\pi^i_k} (\theta^i) = \mathbb{E}_{s\sim \mathcal{D}, a^i_k \sim \pi^{i}_{\theta^i, k}, \varepsilon \sim \mathcal{N}} \left[ \log \pi^i_{\theta^i, k} (f_{\theta^i}(\varepsilon ; s) | s) - Q^i_{\omega^i}(s, f_{\theta^i}(\varepsilon ; s)) \right]
$$

---

The authors note that calculating an expectation, as $k$ increases the variance increases throughout training. To prevent this, the authors uses approximate best response in form of deterministic strategy throughout recursive rollouts. It is simple for discrete action and for Guassian policy the mean is used.

With the approximation, the author mitigate the effect by auxillary loss as
$$
\mathcal{J}_{\pi^i_{\tilde{k}}}(\theta^i) = \mathbb{E}_{s \sim \mathcal{D}^i, (a^i_{\tilde{k}}, a^{-i}_{\tilde{k}}) \sim \pi^i_{\theta^i}, \rho^{-i}_{\phi^{-i}}} \left[ Q^i(s, a^i_{\tilde{k}}, a^i_{\tilde{k}-1}) - Q^i(s, a^i_{\tilde{k}-2}, a^{-i}_{\tilde{k}-1}) \right]
$$
This enforces the higher-level policies to be weakly dominant over lower-level policies (see _Collary_ in theoreical section)



## Theoretical Guarantee

**Theorem 1**: In 2-player, 2-action game, if there exists mixed strategy equilibrium, under mild conditions, the learning dynamics of GR2 methods to the equilibrium is asymptotic statble in sense of Lyapunov.



**Theorem 2**: GR2 strategies extends a norm form game into extensive form games and there exists a perfect Baysian equilibrium in that game.



**Proposition**: In both GR2-L and GR2-mixed if the agent play pure strategy, once level-k agent reaches a Nash equilibrium, all higher agents will follows too.



**Corollary**: In the GR2 setting, higher level strategies weakly dominat lower-level strategy.

*Cover by [@chuttersnap](https://unsplash.com/@chuttersnap) Thanks*
