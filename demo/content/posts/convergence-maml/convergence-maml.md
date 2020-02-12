---
title: "On the Convergence Theory of Gradient-Based Model-Agnostic Meta-Learning Algorithms"
path: "/convergence-maml"
tags: ["Paper Note"]
featuredImage: "./cover.jpg"
excerpt: Paper summery about the convergence of MAML and newly proposed algorithms.
created: 2019-12-08
updated: 2019-12-08
---

Paper URL: <a href="https://arxiv.org/abs/1908.10400">https://arxiv.org/abs/1908.10400</a> by Alireza Fallah, Aryan Mokhtari, Asuman Ozdaglar

# TLDR;
In this paper, the authors analyse Model Agnostic Meta Learning (MAML) from [this paper](https://arxiv.org/abs/1703.03400),
the analysis includes the original MAML algorithms, and its first order approximation. They concluded that MAML converge, however, its first order approximation doesn't in general cases unless some condition applies. Furthermore, they proposed Hessian-Free MAML that is faster to compute (since it doesn't require Hessian computation) compare to MAML but still converge with similar rate.


_Todo_

1. Read the proof.



## MAML Objective

The goal for MAML is to find the initial parameter that can be trained with a few iterations, so that the trained parameter will give a high accuracy. We will assume a finite set of tasks that MAML should cover denoted as

$$
\mathcal{T} = \{ \tau_i \}_{i \in I}
$$

In which, each task $\tau_i$ is sampled from the probability distribution of $P(\tau_i)$. We also denote a loss function $f_i(\boldsymbol{w}) : \mathbb{R}^n \rightarrow \mathbb{R}$. For a normal machine learning task, the optimization objective is then

$$
f(\boldsymbol{w}) = \mathbb{E}_{i \sim P} \left[ f_i(\boldsymbol{w}) \right]
$$

Noted that, for individual task instance, which contains problems that can be specified by $\theta$, thus the loss function $f_i$ is

$$
f_i(\boldsymbol{w}) = \mathbb{E}_{\theta} \left[ f_i(\boldsymbol{w}, \theta) \right]
$$

It is hard to compute the gradient and hessian, due to the number of training data. Therefore, we can estimate the gradient and hessian with a batch $\mathcal{D}$.

$$
\tilde{\nabla} f_i(\boldsymbol{w}, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{\theta \in \mathcal{D}} \nabla f_i(\boldsymbol{w}, \theta)
$$

It is clear that this is unbiased estimate of $\nabla f_i(\boldsymbol{w})$. For MAML, the objective can be equal to

$$
F(\boldsymbol{w}) = \mathbb{E}_{i \sim P} \left[ f_i\left( \boldsymbol{w} - \alpha\nabla_{\boldsymbol{w}} f_i(\boldsymbol{w}) \right) \right]
$$

Given a large dataset, we might need to have unbiased estimate of this following objective denoted as $\hat{F}$

$$
\hat{F}(\boldsymbol{w}) = \mathbb{E}_{i \sim P} \left[ \mathbb{E}_{\mathcal{D}^i_{\text{test}}} \left[ f_i(\boldsymbol{w} - \tilde{\nabla} f_i(\boldsymbol{w}, \mathcal{D}^i_{\text{test}})) \right] \right]
$$

## MAML Algorithm

To optimize the following object, we need 2 loop: Inner (training specialized weight) and Outer (training general initial weight).
For inner loop, for each task $\tau_i$, we use a dataset $\mathcal{D}^i_{\text{in}}$ to compute stochastic gradient, and update task $i$'s specialized weights; at iteration $k

$$
\boldsymbol{w}_{k+1}^i = \boldsymbol{w}_{k}^i - \alpha \tilde{\nabla} f_i(\boldsymbol{w}_{k}^i, \mathcal{D}^i_{\text{in}})
$$

After getting a list of specialized weights for $B$ number of tasks $\{\boldsymbol{w}^i_{k+1}\}^B_{i=1}$. We train the general initial weight as:

$$
\boldsymbol{w}_{k+1} = \boldsymbol{w}_{k} - \beta_k \frac{1}{B} \sum^B_{i=1} \left(\boldsymbol{I} - \alpha \tilde{\nabla}^2 f_i(\boldsymbol{w}_{k}, \mathcal{D}^i_{\text{h}}) \right) \tilde{\nabla} f_i(\boldsymbol{w}_{k+1}^i, \mathcal{D}^i_{\text{o}})
$$

We can see that the complexity of computing Hessian is $\mathcal{O}(d^2 B D_\text{h})$ where $D_\text{h}$ is the size of batch for Hessian approximation and $d$ is dimension of the problem.



## MAML First Order Approximation

To avoid computing Hessian, we can simply remove it, getting First order MAML.

$$
\boldsymbol{w}_{k+1} = \boldsymbol{w}_{k} - \beta_k \frac{1}{B} \sum^B_{i=1} \tilde{\nabla} f_i(\boldsymbol{w}_{k+1}^i, \mathcal{D}^i_{\text{o}})
$$

# Results

## Hessian Free MAML

Noted that the Hessian vector product of any function $\phi$ with a vector $\boldsymbol{v}$ can be approximated by:

$$
\nabla^2 \phi(\boldsymbol{w}) \boldsymbol{v} \approx \left[ \frac{\nabla\phi(\boldsymbol{w} + \delta \boldsymbol{v}) - \nabla\phi(\boldsymbol{w} - \delta \boldsymbol{v})}{2\delta} \right]
$$

The approximation error $\rho \delta ||\boldsymbol{v}||^2$, where $\rho$ is parameter for Lipschitz continuity of Hessian. Then the update of HF-MAML is

$$
\boldsymbol{w}_{k+1} = \boldsymbol{w}_k - \beta_k \frac{1}{B} \sum^B_{i=1} \left[ \tilde{\nabla} f_i(\boldsymbol{w}^i_{k+1}, \mathcal{D}^i_{\text{o}}) - \alpha d^i_k \right]
$$

where $d^i_k$ is defined as:

$$
d^i_k = \frac{\tilde{\nabla} f_i(\boldsymbol{w}_k + \delta^i_k \tilde{\nabla}f_i(\boldsymbol{w}^i_{k+1}, \mathcal{D}^i_{\text{o}}), \mathcal{D}^i_{\text{h}}) - f_i(\boldsymbol{w}_k - \delta^i_k \tilde{\nabla}f_i(\boldsymbol{w}^i_{k+1}, \mathcal{D}^i_{\text{o}}), \mathcal{D}^i_{\text{h}})}{2\delta^i_k}
$$

It is clear that $d^i_k \approx \nabla^2 f_i(\boldsymbol{w}_k, \mathcal{D}^i_{\text{in}}) \tilde{\nabla} f_i(\boldsymbol{w}^i_{k+1}, \mathcal{D}^i_{\text{o}})$ and $\delta^i_k > 0$

**Definition**: A random variable $\omega_{\varepsilon} \in \mathbb{R}^d$ is an $\varepsilon$-approximate first order stationary point (FOSP) is it satisfies
$$
\mathbb{E}\left[ \| \nabla F(\omega_{\varepsilon}) \| \right] \le \varepsilon
$$
The authors would like to see whether MAML algorithms able to find $\varepsilon$-FOSP for any $\varepsilon $ ? If yes, how many iteration is needed.  



# Assumptions

Before making any analysis, the authors state some assumptions regarding the analysis.

- **Assumption 1**: Function $F$ is bounded below:
  $$
  \min_{\omega \in \mathbb{R}^d} F(\omega) > -\infty
  $$

- **Assumption 2**: For every $i \in \mathcal{I}$, $f_i$ is twice differentiable and $L_i$-smooth i.e for all $w, u \in \mathbb{R}^d$ we have
  $$
  \| \nabla f_i(w) - \nabla f_i(u) \| \le L_i\|w - u\|
  $$
  We will denote $L = \max_i L_i$

- **Assumption 3**: For every $i \in \mathcal{I}$, the Hessian of function $f_i$ is $\rho_i$-Lipschitz continuous i.e for all $w, u \in \mathbb{R}^d$ we have:
  $$
  \| \nabla^2 f_i(w) - \nabla^2 f_i(u) \| \le \rho_i\|w - u\|
  $$
  We will denote $\rho = \max_i \rho_i$

- **Assumption 4**: During training, the tasks should be somewhat related, therefore, the authors assume that the variance of $\nabla f(w) = \mathbb{E}_{i \sim p} [\nabla f_i(w)]$ to be bounded. For any $w \in \mathbb{R}^d$, for some non-negative $\sigma$
  $$
  \mathbb{E}_{i \sim p} \left[ \| \nabla f(w) - \nabla f_i(w) \|^2 \right] \le \sigma^2
  $$

- **Assumption 5**: When calculate the gradient, we will have to rely on SGD and therefore, the variance of stochatic gradient $\nabla f_i(w, \theta)$ and Hessian $\nabla^2 f_i(w, \theta)$ should be bounded. For any $i \in \mathcal{I}$  $w \in \mathbb{R}^d$ and non-negative $\tilde{\sigma}, \sigma_H$
  $$
  \begin{aligned}
  &\mathbb{E}_{\theta} \left[\| \nabla f_i(w, \theta) - \nabla f_i(w) \| ^ 2\right] \le \tilde{\sigma}^2 \\
  &\mathbb{E}_{\theta} \left[\| \nabla^2 f_i(w, \theta) - \nabla^2 f_i(w) \| ^ 2\right] \le \sigma^2_H
  \end{aligned}
  $$



# Challenges and Immediate results

There are several challenges that the author face during the analysis of the algorithms, which leads to some immediate results notably:

- Unbounded smooth paramter.
- Stochastic step size
- Biased Estimator
- Connection between $F(w)$ and $\hat{F}(w)$



## Unbounded smooth paramter

Although the individual loss $f_i$ is smooth, the global loss isn't necessary smooth, and can be unbounded as shown in the lemma

**Lemma**: Consider the objective function $F$ and $\alpha \in [0, 1]$ we have
$$
\| \nabla F(w) - \nabla F(w) \| \le \left(4L + 2\rho  \alpha \min \Big\{ \mathbb{E}_{i\sim p} [\|\nabla f_i(w)\|] ,  \mathbb{E}_{i\sim p} [\|\nabla f_i(w)\|] \Big\}\right) \|w - u\|
$$
It is clear that the minimum term can be unbounded



**Corollary**: Extending the smoothness result, we have
$$
-\frac{L(w)}{2} \|u-w\|^2 \le F(u) - F(w) - \nabla F(w)^T (u-v) \le -\frac{L(w)}{2} \|u-w\|^2
$$
where $L(w) = 4L + 2\rho\alpha \mathbb{E}_{i\sim p}[\|\nabla f_i(w)\|]$



## Stochastic step size

Usually, the step size is choosen inversely propotion to smoothness parameter to get the best result. The authors poposed method for choosing the step size by approximate $L(w)$ with average over a batch of tasks as $\tilde{L}(w)$ and we can set the step size to $\beta (w) = c/\tilde{L}(w) $.

The author provided the lower bound of first moment and upper bound of second moment for this approximation as shown in the lemma:



**Lemma**: Consider the objective function $F$, for the case that $\alpha \in [0, \frac{1}{L}]$ suppos the conditions in the assumptions are satisfied the
$$
\frac{1}{\tilde{L}(w)} = \frac{1}{4L + 2\rho\alpha \sum_{j \in \mathcal{B'}} \|\tilde{\nabla} f_j(w, \mathcal{D}^j_\beta)\| /B'}
$$
where $B'$ is the size of $\mathcal{B}'$, and $D^j_\beta$ is dataset of task $i$ with size $D_{\beta}$ if the conditions are satisfied
$$
B' \ge \left\lceil \frac{1}{2} \left( \frac{\rho\alpha\sigma}{L} \right)^2 \right\rceil \quad \quad B_{\beta} \ge \left\lceil \left(\frac{2\rho \alpha \tilde{\sigma}}{L}\right)^2 \right\rceil
$$
Then
$$
\mathbb{E}\left[ \frac{1}{\tilde{L}(w)}\right] \ge \frac{0.8}{L(w)} \quad\quad\quad  \mathbb{E}\left[\left(  \frac{1}{\tilde{L}(w)}\right)^2\right] \le \frac{3.125}{L(w)^2}
$$



## Biased Estimator

Recall that the update rule, the descent direction $g_k$ for MAML at step $k$ is given by
$$
g_k = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \left( I - \alpha \tilde{\nabla}^2 f_i(w_k, \mathcal{D}^i_{in}) \right) \tilde{\nabla}f_i (w_k - \alpha \tilde{\nabla}f_i(w_k, \mathcal{D}^i_{in}), \mathcal{D}^i_o)
$$
Where the exact gradient of $F$ is:
$$
\nabla F(w_k) = \mathbb{E}_{i \sim p} [(I - \alpha \nabla^2 f_i(w_k)) \nabla f_i(w_k - \alpha \nabla f_i(w_k))]
$$
It is clear that the estimator is biased, and we, therefore, have to characterized its first-order and second-order moment.



**Lemma**: Consider the objective function $F$ for $\alpha \in \left[0, \frac{1}{L}\right]$ suppose that the condition in assumptions are tru then, we can show that
$$
\mathbb{E}_{\mathcal{D}_{in}, \mathcal{D}_o} \left[ \tilde{\nabla} f_i(w_k - \alpha \tilde{\nabla} f_i (w_k, \mathcal{D}^i_{in}), \mathcal{D}^i_o) | F_k \right] = \nabla f_i(w_k - \alpha \nabla f_i(w_k)) + e_{i, k}
$$
where $\|e_{i, k}\| \le \frac{\alpha L \tilde{\sigma}}{\sqrt{D_{in}}}$. For the second moment we have
$$
\begin{aligned}
\mathbb{E}&_{\mathcal{D}_{in}, \mathcal{D}_o} \left[ \|\tilde{\nabla} f_i(w_k - \alpha
\tilde{\nabla} f_i (w_k, \mathcal{D}^i_{in}), \mathcal{D}^i_o)\| ^2 | F_k \right] \\
&\le \left( 1 + \frac{1}{\phi} \right) \| f_i(w_k - \alpha \nabla f_i(w_k)) \|^2 + \frac{(1 + \phi) \alpha^2 L^2 \tilde{\sigma}^2}{D_{in}} + \frac{\tilde{\sigma}^2}{D_o}
\end{aligned}
$$
where $\phi$ is arbitrary positive constant.



## Connection between $F$ and $\hat{F}$

We show that all methods get same level of gradient norm with respect to both $F$ and $\hat{F}$ up to a constant.



**Theorem**: Consider the objective function $F$ and $\hat{F}$ for the case that $\alpha \in \left(0, \frac{1}{L}\right]$ and suppose that the conditions in assuptions are satistifed, we have
$$
\| \nabla \hat{F}(w) - \nabla F(w) \| \le 2\alpha L \frac{\tilde{\sigma}}{\sqrt{D_{test}}} + \alpha^2 L \frac{\sigma_H \tilde{\sigma}}{D_{test}}
$$



# Results

Now, after exploring the challenges and intermediate results, the authors show the result for convergence in following algorithms

1. Normal MAML
2. First order MAML
3. Hessian Free MAML

And, we will assume the assumptions to be true.



**Theorem**: Consider the objective function $F$ for the case that $\alpha \in \left( 0, \frac{1}{L} \right]$. Consider running MAML with batch size of $D_h \ge \left\lceil 2\alpha^2\sigma_H^2 \right\rceil$ and $B \ge 20$. Let $\beta_k = \frac{1}{12 \tilde{L}(w_k)}.$ Then for any $\varepsilon > 0$, MAML finds a solution $w_\varepsilon$ such that
$$
\mathbb{E}\left[\| \nabla F(w_\varepsilon) \|\right] \le \max\left\{ \sqrt{61\left( 1 + \frac{\rho\alpha}{L}\sigma \right)\left( \frac{\sigma^2}{B} + \frac{\tilde{\sigma}^2}{BD_o} + \frac{\tilde{\sigma}^2}{D_{\text{in}}} \right)}, \frac{61\rho\alpha}{L} \left( \frac{\sigma^2}{B} + \frac{\tilde{\sigma}^2}{BD_o} + \frac{\tilde{\sigma}^2}{D_{\text{in}}}\right), \varepsilon \right\}
$$
After at most running for
$$
\mathcal{O}(1) \Delta \min \left\{ \frac{L + \rho \alpha (\sigma + \varepsilon)}{\varepsilon^2}, \frac{LB}{\sigma^2} + \frac{L(BD_o + D_{\text{in}})}{\tilde{\sigma}^2} \right\}
$$
where $\Delta = (F(w_0) - \min_{w\in\mathbb{R}^d} F(w))$



**Theorem**: Consider the objective function $F$ for the case that $\alpha \in \left( 0, \frac{1}{10L} \right]$. Consider running FO-MAML with batch size of $D_h \ge \left\lceil 2\alpha^2\sigma_H^2 \right\rceil$ and $B \ge 20$. Let $\beta_k = \frac{1}{18 \tilde{L}(w_k)}.$ Then for any $\varepsilon > 0$, FO-MAML finds a solution $w_\varepsilon$ such that
$$
\begin{aligned}
  \mathbb{E}\left[\| \nabla F(w_\varepsilon) \|\right] \le \max\Bigg\{ &\sqrt{14\left( 1 + \frac{\rho\alpha}{L}\sigma \right)\left( \sigma^2\left( \frac{1}{B} + 20\alpha^2L^2 \right) + \frac{\sigma^2}{BD_o} + \frac{\tilde{\sigma}^2}{D_{\text{in}}} \right)} , \\
  &\frac{14\rho\alpha}{L} \left( \sigma^2 \left( \frac{1}{B} + 20\alpha^2L^2 \right) + \frac{\tilde{\sigma}^2}{BD_o} + \frac{\sigma^2}{D_{\text{in}}} \right), \varepsilon  \Bigg\}
\end{aligned}
$$
After at most running for
$$
\mathcal{O}(1)\Delta \min \left\{ \frac{L + \rho\alpha(\sigma + \varepsilon)}{\varepsilon^2}, \frac{L}{\sigma^2(1/B + 20 \sigma^2 L^2)} + \frac{L(BD_o + D_{\text{in}})}{\tilde{\sigma}^2} \right\}
$$
where $\Delta = (F(w_0) - \min_{w\in\mathbb{R}^d} F(w))$



**Theorem**: Consider the objective function $F$ for the case that $\alpha \in \left( 0, \frac{1}{6L} \right]$. Consider running FO-MAML with batch size of $D_h \ge \left\lceil 36(\alpha\rho\sigma_H)^2 \right\rceil$ and $B \ge 20$. Let $\beta_k = \frac{1}{25 \tilde{L}(w_k)}.$ where the approximation paramter is equal to
$$
\delta^i_k = \frac{1}{6\rho\alpha\| \tilde{\nabla} f_i(w_k - \alpha \tilde{\nabla} f_i(w_k, D^i_{\text{in}}), D^i_o) \|}
$$
Then for any $\varepsilon > 0$, FO-MAML finds a solution $w_\varepsilon$ such that
$$
\mathbb{E}\left[\| \nabla F(w_\varepsilon) \|\right] \le \max\left\{ 6\sqrt{\left( 1 + \frac{\rho\alpha}{L}\sigma \right)\left( \frac{\sigma^2}{B} + \frac{\tilde{\sigma}^2}{BD_o} + \frac{\tilde{\sigma}^2}{D_{\text{in}}} \right)}, 36 \frac{\rho\alpha}{L}\left( \frac{\sigma^2}{B} + \frac{\tilde{\sigma}^2}{BD_o} + \frac{\tilde{\sigma}^2}{D_{\text{in}}} \right), \varepsilon \right\}
$$
After at most running for
$$
\mathcal{O}(1) \Delta \min \left\{ \frac{L + \rho \alpha (\sigma + \varepsilon)}{\varepsilon^2}, \frac{LB}{\sigma^2} + \frac{L(BD_o + D_{\text{in}})}{\tilde{\sigma}^2} \right\}
$$
where $\Delta = (F(w_0) - \min_{w\in\mathbb{R}^d} F(w))$


*Cover by [@seanpaulkinnear](https://unsplash.com/@seanpaulkinnear) Thanks*
