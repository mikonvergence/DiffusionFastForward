# Diffusion Theory

> [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/01-Diffusion-Sandbox-colab.ipynb) [01-Diffusion-Sandbox](https://github.com/mikonvergence/DiffusionFastForward/blob/master/01-Diffusion-Sandbox.ipynb)  - This note relates to the notebook, which includes more  visualizations of the diffusion process!

Diffusion models solve a task similar to GANs (and other generative model types, like VAEs or Normalizing Flows) - they attempt to approximate some probability distribution of a given domain $q(x)$ and **most importantly, provide a way to sample from that distribution** $x \sim q(x)$.

This is achieved by optimizing some parameters $\theta$ (represented as a neural network) that result in a probability distribution $p_\theta(x)$. The objective of training is so that $p_\theta$ produces similar samples $x$ to those drawn from the true underlying distribution $q(x)$.

## :bulb: What's different from GANs?
* GANs produce samples from a latent vector in a single forward pass through the Generator network. The likelihood of the produced samples is controlled by the Discriminator network, which is trained to distinguish between $x \sim q(x)$ and $x \sim p_\theta(x)$.
* Diffusion models use a single network that is used to sequentially converge to an approximation of a real sample $x \sim q(x)$ through several estimation steps. So, the model input and output are generally of the same dimensionality.

## :wrench: Mechanism of the Denoising Diffusion Process
Denoising Diffusion Process consists of a chain of steps in two directions, corresponding do destruction and creation of information in the sample.

### :point_right: Forward Process
With access to a sample at a time step $t$, one can make an estimation about the next sample in the forward process, defined by the true distribution $q$:

```math
q(x_{t}|x_{t-1})\tag{1}
```

Quite often, what is available are the samples at time step $0$ (meaning clean samples), and it is then useful to use the types of operation that allow easy and efficient formulation of:

```math
q(x_{t}|x_{0})\tag{2}
```

So far, the **most common** choice for a forward process has been **Gaussian**. Easy to compute and convenient in various respects:

```math
q(x_{t}|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)\tag{3}
```

the notation above simply means that the **previous sample is scaled down** by a factor of $\sqrt{1-\beta_t}$ and **additional Gaussian noise** (sampled from a zero-mean unit-variance Gaussian) multiplied by $\beta_t$ **is added**.

Furthermore, the $0\to t$ step can also be easily defined as:

```math
q(x_{t}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t}) I) \tag{4}
```

where $\alpha_t = 1-\beta_t$ and

```math
\bar{\alpha_t}=\prod_{i=0}^{t}\alpha_t \tag{5}
```

### :point_left: Reverse Process
The reverse process is designed to restore the information in the sample, which allows to generate a new sample from the distribution. Generally, it will start at some high time step $t$ (very often at $t=T$, which indicates the end of the diffusion chain, where the probability distribution is extremely close to a pure Gaussian), and attempt to approximate the distribution of the previous sample $t-1$.

```math
p_\theta(x_{t-1}|x_t)
```

If diffusion steps are small enough, the reverse process of a Gaussian forward process can also be approximated by a Gaussian:

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))\tag{6}
```

The reverse process is often parameterized using a neural network $\theta$, a common good candidate for approximating complex transformations. In many cases, a standard deviation function $\sigma_t$ independent of $x_t$ can be used:

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\sigma_t^2 I)\tag{7}
```

## :steam_locomotive: DDPM: Denoising Diffusion Probabilistic Model

[DDPM](https://arxiv.org/abs/2006.11239) is one of the first popular approaches to denoising diffusion. It generates samples by following the reverse process through all *T* steps of the diffusion chain.

When it comes to parameterizing the mean $\mu_\theta(x_t,t)$ of the reverse process distribution, the network can either:
1. Predict it directly as $\mu_\theta(x_t,t)$
2. Predict the original $t=0$ sample $x_0$, where

```math
\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \tag{8}
```

3. Predict the **normal** noise sample $\epsilon$ (from a unit-variance distribution), which has been added to the sample $x_0$

```math
x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon) \tag{9}
```

The third option, where the network predicts $\epsilon$ appears to be most common, and that's what is being done in DDPM. This yields to a new equation for $\tilde{\mu}_{\theta}$ expressed in terms of $x_t$ and $\epsilon$:


```math
\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}(\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \tag{10}
```

and hence

```math
\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon) \tag{11}
```

...which is the key equation for DDPM used for sampling.

### Training
Training a model tasked to predict the noise shape $\epsilon$ is quite straightforward.

At each training step:
1. Use **forward process** to generate a sample $x_t \sim q(x_t|x_0)$ for a $t$ sampled uniformly from $[1,T]$:
    1. Sample time step $t$ from a uniform distribution $t \sim \mathcal{U}(1,T)$
    2. Sample $\epsilon$ from a normal Gaussian $\epsilon \sim \mathcal{N}(0,1)$
    3. Compute noisy input sample $x_t$ for training via $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
3. **Compute the approximation of noise** $\hat{\epsilon_t}=p_\theta(x_t,t)$ using the model with parameters $\theta$
4. **Minimize the error** between $\epsilon_t$ and $\hat{\epsilon_t}$ by optimizing parameters $\theta$

### Sampling
Generation begins at $t=T$ by sampling from the last step $x_T \sim \mathcal{N}(0,1)$ in the diffusion process, which is modelled by a normal Gaussian.

Then, until $t=0$ is reached, the network makes a prediction of noise in the sample $\tilde{\epsilon}=p_\theta(x_t,t)$ and then approximates the mean of the process at $t-1$, using:

```math
\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon})\tag{12}
```

Hence, the next sample at $t-1$ is sampled from the Gaussian distribution like below:

```math
x_{t-1} \sim \mathcal{N}(\tilde{\mu}_\theta,\sigma_t^2 I)\tag{13}
```

...until $x_0$ is reached, in which case only the mean $\tilde{\mu}_\theta$ is extracted as output.

## :bullettrain_front: (Sampling Faster) DDIM: Denoising Diffusion Implicit Model

> Warning: if you look up the original DDIM paper, you will see the symbol $\alpha_t$ used for $\bar{\alpha}_t$. In this note, no such notation change is made for the sake of consistency.

`DDPM` reverse process attempts to navigate the diffusion chain of `T` steps in the reverse order. However, there as shown in (9), the reverse process involves an approximation of the clean sample $x_0$. 

If we substitute $t-1$ for $t$ in (4):

```math
q(x_{t-1}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1}) I)\tag{14}
```

which yields

```math
x_{t-1} \leftarrow \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{t-1}\tag{15}
```

...and based on a specific $\epsilon_t$ measured at the previous step $t$, it can be rewritten as:

```math
x_{t-1} \leftarrow \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \epsilon_{t} + \sigma_t \epsilon\tag{16}
```

Generally, $\sigma_t$ is set to:

```math
\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\tag{17}
```

Further, we can introduce a new parameter $\eta$ to control the magnitude of the stochastic component:

$$\sigma_t^2 = \eta \tilde{\beta}_t \tag{18}$$

As found in the original [DDIM paper](https://arxiv.org/abs/2010.02502), setting $\eta=0$ appears to be particularly beneficial when fewer steps of the reverse process are applied and that specific type of process is known as **Denoising Diffusion Implicit Model (DDIM)**. The above formulation is still consistent with DDPM when $\eta=1$.

:flashlight: So, how can the reverse chain be navigated in the reverse direction? First, a sequence of fewer steps $S$ is defined as a subset $\{\tau_1, \tau_2, ..., \tau_S\}$ of the original temporal steps of the forward process. Sampling is then based on (16).

At each step:
1. Predict $x_0$
2. Compute the direction towards current $x_t$
3. (If not DDIM) inject some noise for the stochastic functionality

It can generally be assumed that DDIM
* Offers better sample quality at fewer steps
* Allows for deterministic matching between the starting noise $x_T$ and the generated sample $x_0$
* Performs worse than DDPM for large numbers of steps (such as 1000)
