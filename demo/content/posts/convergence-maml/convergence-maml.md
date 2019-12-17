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

## Background

We will started by introducing the objective of MAML and what it tries to achieved. Then we turn to the optimization algorithms and its approximation.  

### MAML Objective

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


### MAML Algorithm

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

### MAML First Order Approximation

To avoid computing Hessian, we can simply remove it, getting First order MAML.

$$
\boldsymbol{w}_{k+1} = \boldsymbol{w}_{k} - \beta_k \frac{1}{B} \sum^B_{i=1} \tilde{\nabla} f_i(\boldsymbol{w}_{k+1}^i, \mathcal{D}^i_{\text{o}})
$$


## Results

### Hessian Free MAML

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

### Theoretical Analysis of MAML


*Cover by [@seanpaulkinnear](https://unsplash.com/@seanpaulkinnear) Thanks*
