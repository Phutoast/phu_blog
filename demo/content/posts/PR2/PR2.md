---
title: "Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning"
path: "/PR2"
tags: ["Paper Note"]
featuredImage: "./cover.jpg"
excerpt: Paper summery of PR2
created: 2019-02-07
updated: 2019-02-07
---

Paper URL: <a href="https://arxiv.org/abs/1901.09207">https://arxiv.org/abs/1901.09207</a> by Ying Wen, Yaodong Yang, Rui Luo, Jun Wang, Wei Pan

# Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning

_TODO_

- Redo all the proofs -- so that i matched the same patterns.
- Do the algorithms



## Formulations

The author starts off with the way of factorizing the probability distribution $\pi_{\theta}(a^i_t,  a^{-i}_t| s_t)$ , which leads to so-called "recursive reasoning".
$$
\pi_{\theta}(a^i_t,  a^{-i}_t| s_t) = \pi_{\theta}^i(a^i_t | s_t) \cdot \pi_{\theta}^{-i}(a^{-i}_t | s_t, a^i_t)
$$
Furthermore, for each agent, it can have its own opponent model $\rho_{\phi}(a^{-i}_t | s_t, a_t^i)$. With this, one can define the following optimization scheme
$$
\begin{aligned}
	&\arg\max_{\theta^{i}, \phi^{i}} \eta_i\left( \pi^i_{\theta^{i}}(a^i| s) \rho^{-i}_{\phi^{i}}(a^{-i} | s, a^i) \right) \\
	&\arg\max_{\theta^{-i}, \phi^{-i}} \eta_i\left( \pi^i_{\theta^{-i}}(a^i| s) \rho^{-i}_{\phi^{-i}}(a^{-i} | s, a^i) \right)
\end{aligned}
$$
Where $\eta$ is the expected future-discounted reward. By using the policy gradient theorem and factorization above, the authors showed that
$$
\nabla_{\theta^i} \eta^i  = \mathbb{E}_{s\sim p_s, a^i \sim \pi^i} \left[ \nabla_{\theta^i} \log \pi^i_{\theta^i} (a^i | s) \int_{a^{-i}} \pi^{-i}_{\theta^{-i}} (a^{-i} | s, a^i)Q^i(s, a^i, a^{-i} ) \ da^{-i} \right]
$$
This is quite intuitive because we would like to maximizes future expected reward by marginalizing all the opponent actions. With the importance sampling one and get unbiased estimate of the integral term.
$$
\nabla_{\theta^i} \eta^i  = \mathbb{E}_{s\sim p_s, a^i \sim \pi^i} \left[ \nabla_{\theta^i} \log \pi^i_{\theta^i} (a^i | s) \ \mathbb{E}_{a^{-i} \sim \rho^{-i}_{\phi}}\left[ \frac{\pi^{-i}_{\theta^{-i}} (a^{-i} | s, a^i)}{\rho^{-i}_{\phi}(a^{-i} | s, a^i)} Q^i(s, a^i, a^{-i} ) \right] \right]
$$

Now, we turned into variational inference, we defined the optimal trajectory to be
$$
P(\tau) = P(s_1) \prod^T_{t=1} P(s_{t+1} | s_t, a_t) \exp(r(s_t, a^i_t, a^{-i}_t))
$$
With the variational distribution that we used to approximate the optimal trajectory as
$$
q(\tau) = P(s_1) \prod^T_{t=1} P(s_{t+1} | s_t, a_t) \pi(a^i_t | s_t) \rho(a^{-i}_t | s_t, a^i_t)
$$
Now, one can minimize the KL-divergence between $P(\tau)$ and $q(\tau)$ and optimizing for both $\pi_{\theta}(a^i_t | s_t)$ and $\rho_{\phi}(a^{-i} | a^i_t | s_t)$ which are policy and opponent model, respectively
$$
D_{KL}\left( P(\tau) \Big\| q(\tau) \right) = -\sum^T_{t=1} \mathbb{E}_{\tau \sim q(\tau)}\left[ r(s_t, a^i_t, a^{-i}_t) + \mathcal{H}(\pi(a^i_t | s_t) \rho(a^{-i}_t | s_t, a^i_t)) \right]
$$
We can derived closed form solution for the opponent model, which is
$$
\rho(a^{-i}_t | s_t, a^i_t) = \frac{\exp\left( Q^i(s_t, a^{i}_t, a^{-i}_t) - Q^i (s_t, a^i_t) \right)}{Z(s_t, a_t^i, a^{-i)}_t)} \quad \text{ where }  Q^i (s_t, a^i_t)  = \int_{a^{-i}} \exp(Q^i(s_t, a^i_t, a^{-i}) \ da^{-i}
$$


## Implementation

The authors follow the implementation of soft-Q learning, which uses SVGD, which samples the opponent model, with this the agent can also find its expected future reward given opponent's model
$$
\int_{a^{-i}}  \pi^{-i}_{\theta^{-i}}(a^{-i}_t | s_t, a^i_t) Q^i(s_t, a^i_t, a^{-i}) \ d a^{-i}_t
$$



## Theoretical Guarantee

The authors has also provided a guarantee of convergence of the contraction mapping
$$
\mathcal{T} Q(s_t, a^i_t, a^{-i}_t) = r^i(s_t, a_t^i, a_{t}^{-i}) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1})}\left[ \log \int_{a^{-i}} \exp\left( Q(s_{t+1}, a_{t+1}^i, a^{-i}_{t+1}) \right) \ da^{-i} \right]
$$
given the following assumptions

1. Symmetric game
2. One global optimum $\mathbb{E}_{\pi^*}[Q^i(s)] \ge \mathbb{E}_{\pi}[Q^i(s)]$ OR saddle point $\mathbb{E}_{\pi^*}[Q^i(s)] \ge \mathbb{E}_{\pi^i}\mathbb{E}_{\pi^{-i *}}[Q^i(s)]$ or $\mathbb{E}_{\pi^*}[Q^i(s)] \ge \mathbb{E}_{\pi^{i *}}\mathbb{E}_{\pi^{-i }}[Q^i(s)]$

*Cover by [@cadop](https://unsplash.com/@cadop) Thanks*
