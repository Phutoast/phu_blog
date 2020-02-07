---
title: "A Regularized Opponent Model with Maximum Entropy Objective"
path: "/ROMMEO"
tags: ["Paper Note"]
featuredImage: "./cover.jpg"
excerpt: Paper summery of ROMMEO
created: 2019-02-07
updated: 2019-02-07
---

Paper URL: <a href="https://arxiv.org/abs/1905.08087">https://arxiv.org/abs/1905.08087</a> by Zheng Tian, Ying Wen, Zhichen Gong, Faiz Punakkath, Shihao Zou, Jun Wang

# ROMMEO (A Regularized Opponent Model with Maximum Entropy Objective)

_TODO_

- Redo all the proofs -- so that i matched the same patterns.
- Understand the opponent model loss 😎
- Do the algorithms

## Problem formulation

The authors define optimum strategy for $n$ agents $(\pi^{*1}, \cdots, \pi^{*n})$ as
$$
\mathbb{E}_{s\sim p_s, a^{i*}_t \sim \pi^{i*}, a^{-i *}_{t} \sim \pi^{-i *}} \left[ \sum^{\infty}_{t=1} \gamma^t R^i(s_t,  a^{i*}_t ,  a^{-i*}_t ) \right] \ge \mathbb{E}_{s\sim p_s, a^{i}_t \sim \pi^{i}, a^{-i}_{t} \sim \pi^{-i }}\left[ \sum^{\infty}_{t=1} \gamma^t R^i(s_t,  a^{i}_t ,  a^{-i}_t ) \right]
$$
For all $\pi \in \Pi$ and $i \in (1, \cdots, n)$. This leads to an objective $\mathcal{J}$ as
$$
\max \mathcal{J} = \log P(\mathcal{O}_i = 1 | \mathcal{O}_{-i} = 1)
$$
To solve this, they used variational inference $q(s_{1:T}, a^i_{1:T}, a^{-i}_{1:T} | \mathcal{O}^i_{1:T} = 1, \mathcal{O}^{-i}_{1:T} = 1)$, which can be factorized as
$$
P(\tau) = P(s_1) \prod^T_{t=1} P(s_{t+1} | s_t, a_t) \exp(r(s_t, a_t))
$$


The variational family is defined by
$$
q(\tau) = P(s_1) \prod^T_{t=1} P(s_{t+1} | s_t, a_t) \pi(a^i_t | s_t, a^{-i}_t) \rho(a^{-i}_t | s_t)
$$
With this, we can arrived at the evidence lower bound (ELBO) as follows
$$
\mathbb{E}_{s_t} \left[ \mathbb{E}_{a^i_t \sim \pi, a^{-t}_t \sim \rho} \left[R^i(s_t, a^i_t, a^{-i}_t ) + \mathcal{H}(\pi(a^i_t | a^{-i}_t, s_t))\right] \right] - \mathbb{E}_{a^{-i}_t \sim \rho} \left[ D_{KL}\left( \rho(a^{-i}_t | s_t) \bigg\|  P(a^{-i}_t| s_t)  \right) \right]
$$
Note that $\rho(a^{-i}_t | s_t, \mathcal{O}_{-i} = 1)$ is the agent's optimal opponent model. Furthermore, updating this not only optimizes $\pi$ but also optimizing $\rho$ in which the KL-term will keep the optimal opponent model being more realistic.

---

The author proceeded to define the closed form solutions for both opponent model and agent, which is
$$
\pi^*(a^{i} | s^i, a^{-i}) = \frac{\exp\left( \frac{1}{\alpha} Q^{\pi^*, \rho^*}_{\text{soft}} (s, a^i, a^{-i}) \right)}{\sum_{a^i} \exp \left( \frac{1}{\alpha} Q^{\pi^*, \rho^*}_{\text{soft}} (s, a^i, a^{-i})   \right)} \quad \quad \rho^*(a^{-i} | s) = \frac{P(a^{-i}|s) \left( \sum_{a^i} \exp \left( \frac{1}{\alpha} Q^{\pi^*, \rho^*}_{\text{soft}} (s, a^i, a^{-i})   \right) \right)^\alpha}{\exp V^*_{\text{soft}}(s)}
$$


Where $Q^{\pi^*, \rho^*}_{\text{soft}} (s, a^i, a^{-i})$ is defined as
$$
Q^{\pi^*, \rho^*}_{\text{soft}} (s_t, a^i_t, a^{-i}_t) = r_t  + \mathbb{E}_{q} \left[ \sum^{\infty}_{l=1} \gamma^t \left( r_{l+t} + \alpha \mathcal{H}(\pi^*(a^i_{t+l} | s_{t+l}, a^{-i}_{l +t})) + D_{KL}\left( \rho(a^{-i}_{l+t} | s_{l+t}) \bigg\|  P(a^{-i}_{l+t}| s_{l+t})  \right) \right)  \right]
$$
and $V^*_{\text{soft}}(s)$ is defined as
$$
V^*_{\text{soft}}(s_t) = \log \sum_{a^{-i} }P(a^i_t|s_t) \left( \sum_{a^i_t} \exp \left( \frac{1}{\alpha} Q^{\pi^*, \rho^*}_{\text{soft}} (s_t, a^i_t, a^{-i}_t)   \right) \right)^\alpha
$$
Analogously, the authors shows that this leads to something resemblance to Bellman equation
$$
Q^{\pi, \rho}_{\text{soft}}(s_t, a^i_t, a^{-i}_t) = r_t + \gamma \mathbb{E}_{s+1}\left[ V_{\text{soft}}(s_{t+1}) \right]
$$

## Theoretical Guarantee

After arriving at the _Bellman equation_, the authors show that the soft-value iteration operator $\mathcal{T}$ defined as
$$
\mathcal{T} Q^{\pi, \rho}_{\text{soft}}(s_t, a^i_t, a^{-i}_t) = R(s_t, a^i_t, a^{-i}_t) + \gamma \mathbb{E}_{s_{t+1} \sim p_s} \left[ \log \sum_{a^{-i}_{t+1} }P(a^i_{t+1}|s_{t+1}) \left( \sum_{a^i_{t+1}} \exp \left( \frac{1}{\alpha} Q^{\pi, \rho}_{\text{soft}} (s_{t+1}, a^i_{t+1}, a^{-i}_{t+1})   \right) \right)^\alpha \right]
$$
is a contraction mapping under the following assumptions

1. Symmetric Game
2. One global optimum i.e $E_{\pi^*}[Q^i_t(s)] \ge E_{\pi}[Q^i_t(s)]$
3. Bounded $Q^{\pi, \rho}_{\text{soft}}(s_t, a^i_t, a^{-i}_t)$ and $V_{\text{soft}}(s_t)$
4. $Q^{\pi^*, \rho^*}_{\text{soft}} (s, a^i, a^{-i}) < \infty$ and $\exp V^*_{\text{soft}}(s) < \infty$

## Implementation

The authors follows the implementation details from soft-actor critic, and parameterizing $Q_{\omega}(s, a^i, a^{-i})$, $\pi_{\theta}(a^i | s, a^{-i})$ and $\rho_{\phi}(a^{-i} | s)$ . For Q-function, the objective function is a mean square error with targeted Q value  
$$
\mathcal{J}(\omega) = \mathbb{E}_{(s_t, a_t^i, a^{-i}_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left(Q_{\omega}(s_t, a_t^i, a_t^{-i}) - R(s_t, a^i_t, a^{-i}_t) - \gamma \mathbb{E}_{s_{t+1} \sim p_s}[\overline{V}(s_{t+1})]\right)^2 \right]
$$
where $\overline{V}(s_{t+1})$ is
$$
\overline{V}(s_{t+1}) = Q_{\bar{\omega}}(s_{t+1}, a^i_{t+1}, \hat{a}^{-i}_{t+1}) - \log \rho_{\phi}(\hat{a}^{-i}_{t+1} | s_{t+1}) - \alpha \log \pi_{\theta}(a^i_{t+1} | s_{t+1}, \hat{a}^{-i}_{t+1})  + \log P(\hat{a}^{-i}_{t+1} | s_{t+1})
$$
and $\hat{a}^{-i}_{t+1} \sim \rho(a_{t+1}^{-i} | s_{t+1}) $, a sample from opponent model, and $a^{i}_t$ is the action that agent took. For polic, the authors used minimizing KL-divergence.
$$
\mathcal{J}(\theta)= \mathbb{E}_{s_t \sim \mathcal{D}, a^{-i}_t \sim \rho}\left[ D_{KL}\left( \pi_{\theta}(\cdot | s_t, a^{-i}_t ) \Bigg\| \frac{\exp\left(\frac{1}{\alpha}Q_{\omega}(s_t, \cdot, a^{-i}_t)\right)}{Z_{\omega}(s_t, a^{-i}_t)} \right) \right]
$$
and similarly for the opponent model
$$
\mathcal{J}(\phi) = \mathbb{E}_{(s_t, a_t^i) \sim \mathcal{D}} \left[ D_{KL}\left( \rho_{\phi}(\cdot | s^i_t) \Bigg\| \frac{P(\cdot|s_t) \left( \frac{\exp(\frac{1}{\alpha
}Q(s_t, a_t^i, \cdot))}{\pi_{\theta}(a^i_t | s_t, \cdot)} \right)^\alpha}{Z_{\omega}(s_t)} \right) \right]
$$
It's clear that the normalizing constant isn't depends on $\theta$ and we can use reparameter trick on the policy and opponent model as $\hat{a}^{-i}_t \sim g_{\phi}(\varepsilon_t^{-i}; s_t)$ and $a^i_t \sim f_{\theta}(\varepsilon^i_t ; s_t, a^{-i}_t)$  and so we have the following loss function, for policy (this is just expanding the KL-term)
$$
\mathcal{J}(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}, \varepsilon^i \sim \mathcal{N}, \varepsilon^{-i} \sim \mathcal{N}} \left[ \log \pi_{\theta}(f_{\theta}(\varepsilon^i_t ; s_t, \hat{a}^{-i}_t)) - \frac{1}{\alpha} Q_{\omega}(s_t,f_{\theta}(\varepsilon^i_t ; s_t, \hat{a}^{-i}_t) , \hat{a}^{-i}_t) \right]
$$
and the same methods applied to opponent model

$$
\begin{aligned}
	\mathcal{J}(\phi) = \mathbb{E}_{(s_t, a_t^i) \sim \mathcal{D}, \varepsilon^{i}_t \sim \mathcal{N}} \Bigg[ \log \rho_{\phi}(g_{\phi}(\varepsilon_t^{-i}; s_t)) - \log P(g_{\phi}(\varepsilon_t^{-i}; s_t) | s_t) \\
	- Q(s_t, a_t^i, g_{\phi}(\varepsilon_t^{-i}; s_t)) + \alpha \log \pi_{\theta}(a_t^i | s_t, g_{\phi}(\varepsilon_t^{-i}; s_t)) \Bigg]
\end{aligned}
$$


*Cover by [@joshappel](https://unsplash.com/@joshappel) Thanks*
