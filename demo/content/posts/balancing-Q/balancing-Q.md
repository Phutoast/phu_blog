---
title: "Balancing Two-Player Stochastic Games with Soft Q-Learning"
path: "/balancing-Q"
tags: ["Paper Note"]
featuredImage: "./cover.jpg"
excerpt: Paper summery of Balancing Q.
created: 2019-02-07
updated: 2019-02-07
---

Paper URL: <a href="https://arxiv.org/abs/1802.03216">https://arxiv.org/abs/1802.03216</a> by Jordi Grau-Moya, Felix Leibfried, Haitham Bou-Ammar

# Balancing Two-Player Stochastic Games with Soft Q-Learning



_TODO_

- Redo all the proofs -- so that i matched the same patterns.
- Do the algorithms
- Understand regret proof



## Formulations

The authors don't relies on traditional variational inference, instead, they relied on contrained optimization. For example, in single agent case, we have the following objective
$$
\arg\max_{\pi} \mathbb{E}_{\pi}\left[\sum^\infty_{t=0} \gamma^t r(s, a)\right] \text{ such that } \sum^\infty_{t=0}\mathbb{E}\left[\gamma^t D_{KL}\left( \pi(a_t | s_t) \Big\| \pi_{\operatorname{prior}}(a_t|s_t)  \right)\right] \le C
$$
In 2-player game, the authos has defined the value function (which can also be objective) for both players as following:
$$
V^{\pi\rho}(s) = \mathbb{E}\left[ \sum^\infty_{t=0} \gamma^t \left(r(s, a) - \frac{1}{\beta^i} \log\frac{\pi(a_t|s_t)}{\pi_{\operatorname{prior}}(a_t|s_t)} - \frac{1}{\beta^{-i}}\log \frac{\rho(a_t|s_t)}{\rho_{\operatorname{prior}}(a_t|s_t)} \right) \right]
$$
For 2-player game (stochastic game), the authors made the following observation.

1. For zero-sum games, the optimal value for both agents are
   $$
   V^{\pi^*}(s) = \max_{\pi} \min_{\rho}  V^{\pi \rho}(s) \quad \text{ and } \quad  V^{\rho^*}(s) = \min_{\rho} \max_{\pi}  V^{\pi \rho}(s)
   $$
   and by the result of min-max theorem proved by Von Neumann that $V^{\pi^*}(s) = V^{\rho^*}(s)$

2. For cooperative game, the optimal values for both players are obviously the same
   $$
   V^{\pi^*}(s) = V^{\rho*}(s) = \max_{\pi}\max_{\rho}V^{\pi \rho}(s) = \max_{\rho}\max_{\pi} V^{\pi \rho}(s)
   $$

Therefore, given extreme operator $\operatorname{ext}$ which can be either $\min $ or $ \max$ , we can arrived at the optimal value function as
$$
V^*_{\pi}(s) = \max_\pi\underset{\rho}{\operatorname{ext}} V^{\pi\rho}(s) \quad \text{ and } \quad V^*_{\rho}(s) = \underset{\rho}{\operatorname{ext}}\max_{\pi} V^{\pi\rho}(s)
$$
Now, given these facts, the author presents _free energy_ operators as. Note that $\beta$ represents the rationality of the agent, and its sign represents its perception of the game reward $r(s, a^i, a^{-i})$ if their sign are opposite the game is zero-sum, while if their sign are the same, the game is cooperative.
$$
f(\pi, \rho, s, V) = \mathbb{E}_{\pi\rho}\left[ r(s, a^i, a^{-i}) + \mathbb{E}_{s'}[V(s's)] - \frac{1}{\beta^i} \log\frac{\pi(a_t|s_t)}{\pi_{\operatorname{prior}}(a_t|s_t)} - \frac{1}{\beta^{-i}}\log \frac{\rho(a_t|s_t)}{\rho_{\operatorname{prior}}(a_t|s_t)} \right]
$$
Then the Bellman equation for both players are (player, opponent respectively)
$$
\mathcal{B}_{\pi} V = \max_\pi\underset{\rho}{\operatorname{ext}} f(\pi, \rho, s, V) \quad \text{ and } \quad \mathcal{B}_{\rho} V = \underset{\rho}{\operatorname{ext}}\max_{\pi} f(\pi, \rho, s, V)
$$




## Implementations

Now given the formulations, the authors calculated the closed form solution of the optimal agent $\pi^*$ and opponent $\rho^*$.  Started with the definition of Q-value
$$
Q(s, a^i, a^{-i}) = r(s, a^i, a^{-i}) + \mathbb{E}[V(s')]
$$
Then the Q-value for each agents can be calulated by finding the expected, given the other players
$$
\begin{aligned}
&Q^i(s, a^i) = \frac{\beta^i}{\beta^{-i}} \log \sum_{a^{-i}} \rho_{\operatorname{prior}}(a^{-i} | s) \exp \left( \beta^{-i} Q(s, a^i, a^{-i}) \right) \\
&Q^{-i}(s, a^{-i}) = \frac{\beta^{-i}}{\beta^i} \log \sum_{a^{i}} \pi_{\operatorname{prior}}(a^{i} | s) \exp \left( \beta^{-i} Q(s, a^i, a^{-i}) \right)
\end{aligned}
$$
Finally, the optimal agents are
$$
\pi^*(a^i | s) = \frac{1}{Z^{\pi}(s)}\pi_{\operatorname{prior}}(a^i|s) \exp\left(Q^i(s, a^i)\right) \quad \text{ and } \quad \rho^*(a^{-i} | s) = \frac{1}{Z^{\rho}(s)}\pi_{\operatorname{prior}}(a^{-i}|s) \exp\left(Q^{-i}(s, a^{-i})\right)
$$
The normalizing constant can be calulated if with small discrete actions. Furthermore, the value function is
$$
V^*(s) = \frac{1}{\beta^i} \log \sum_{a^i} \pi_{\operatorname{prior}}(a^i|s) \exp\left(Q^i(s, a^i)\right)
$$
Since the paper assumes small discreate action, the agents can be implemented by directly calculating the closed form solution. However, for Q-function, it can be learnt by the following update rule, on k-th iteration
$$
Q^{k+1}(s, a^i, a^{-i}) = Q^k(s, a^i, a^{-i}) + \alpha\left( r(s, a^i, a^{-i}) + \gamma V(s') - Q^k(s, a^i, a^{-i}) \right)
$$
The advantage of defining the $\beta$ value is that one can approximate it with ease (since it is only 1D). The authors proposed online-maximum likelihood by first collectings the following data at episode $j$ with length $m$
$$
\mathcal{D} = \left\{ s^{(j)}_i , a^{\pi \ (j)}_i, a^{\rho \ (j)}_i \right\}^{m^{(j)}}_{i=1}
$$
Then finding $\beta^{-i}$ being
$$
\beta^{-i}_{\text{approx}} = \underset{\beta^{-i}}{\arg\max} \sum^{m^{(j)}}_{i=1} \log \rho^*(a^{\rho \ (j)}_i | s_i^{(j)})
$$
with $\rho$ is parameterized by $\beta^{-i}$



## Theoretical Guarantee

We can show that
$$
\mathcal{B}_{\pi} V(s) = \mathcal{B}_{\rho}V(s)
$$
This defines the general operator $\mathcal{B}$. The general operator can be proved as $L_{\infty}$ contraction norm or we have
$$
\|\mathcal{B} V - \mathcal{B}\bar{V}\|_{\infty} \le \gamma \|V - \bar{V}\|_{\infty}
$$
For the online-maximum likelihood, the authors also derived the regret bound on the estimation of $\beta$ we have
$$
\sum^R_{j=1} \mathcal{L}_j(\beta^{(j)}) - \min_u\left[\sum^R_{j=1} \mathcal{L}_j(u) \right] \approx \mathcal{O}(\sqrt{R})
$$
With the time-varying opponent, the dynamic regret bounded is equal to
$$
\sum^R_{j=1} \mathcal{L}_j(\beta^{(j)}) - \sum^R_{j=1} \min_{u_j} \mathcal{L}_j(u_j) \approx \mathcal{O}\left( \sqrt{R}  + 1 + \sum^{R-1}_{j=1} \| u_{j+1}^* - u^*_{j} \|^2_2 \right)
$$


*Cover by [@loicleray](https://unsplash.com/@loicleray) Thanks*
