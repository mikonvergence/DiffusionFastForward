# Diffusion Theory

Diffusion models solve a task similar to GANs (and other generative model types, like VAEs or Normalizing Flows) - they attempt to approximate some probability distribution of a given domain $q(x)$ and **most importantly, provide a way to sample from that distribution** $x \sim q(x)$.

This is achieved by optimizing some parameters $\theta$ (represented as a neural network) that result in a probability distribution $p_\theta(x)$. The objective of training is so that $p_\theta$ produces similar samples $x$ to those drawn from the true underlying distribution $q(x)$.

## :bulb: What's different from GANs?
* GANs produce samples from a latent vector in a single forward pass through the Generator network. The likelihood of the produced samples is controlled by the Discriminator network, which is trained to distinguish between $x \sim q(x)$ and $x \sim p_\theta(x)$.
* Diffusion models use a single network that is used to sequentially converge to an approximation of a real sample $x \sim q(x)$ through several estimation steps. So, the model input and output are generally of the same dimensionality.

## :wrench: Mechanism of the Denoising Diffusion Process
Denoising Diffusion Process consists of a chain of steps in two directions, corresponding do destruction and creation of information in the sample.

### :point_right: Forward Process
With access to a sample at a time step $t$, one can make an estimation about the next sample in the forward process, defined by the true distribution $q$:

$$ q(x_{t}|x_{t-1}) $$

Quite often, what is available are the samples at time step $0$ (meaning clean samples), and it is then useful to use the types of operation that allow easy and efficient formulation of:

$$ q(x_{t}|x_{0}) $$

So far, the **most common** choice for a forward process has been **Gaussian**. Easy to compute and convenient in various respects:

$$ q(x_{t}|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I) $$

the notation above simply means that the **previous sample is scaled down** by a factor of $\sqrt{1-\beta_t}$ and **additional Gaussian noise** (sampled from a zero-mean unit-variance Gaussian) multiplied by $\beta_t$ **is added**.

Furthermore, the $0\to t$ step can also be easily defined as:

$$ q(x_{t}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t}) I) $$

where $\alpha_t = 1-\beta_t$ and

$$\bar{\alpha_t}=\prod_{i=0}^{t}\alpha_t$$

### :point_left: Reverse Process
The reverse process is designed to restore the information in the sample, which allows to generate a new sample from the distribution. Generally, it will start at some high time step $t$ (very often at $t=T$, which indicates the end of the diffusion chain, where the probability distribution is extremely close to a pure Gaussian), and attempt to approximate the distribution of the previous sample $t-1$.

$$ p_\theta(x_{t-1}|x_t) $$

If diffusion steps are small enough, the reverse process of a Gaussian forward process can also be approximated by a Gaussian:

$$ p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)) $$

The reverse process is often parameterized using a neural network $\theta$, a common good candidate for approximating complex transformations. In many cases, a standard deviation function $\sigma_t$ independent of $x_t$ can be used:

$$ p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\sigma_t^2 I) $$

## :steam_locomotive: DDPM: Denoising Diffusion Probabilistic Model

[DDPM](https://arxiv.org/abs/2006.11239) is one of the first popular approaches to denoising diffusion. It generates samples by following the reverse process through all *T* steps of the diffusion chain.

When it comes to parameterizing the mean $\mu_\theta(x_t,t)$ of the reverse process distribution, the network can either:
1. Predict it directly as $\mu_\theta(x_t,t)$
2. Predict the original $t=0$ sample $x_0$, where

$$\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

3. Predict the **normal** noise sample $\epsilon$ (from a unit-variance distribution), which has been added to the sample $x_0$

$$x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)$$

The third option, where the network predicts $\epsilon$ appears to be most common, and that's what is being done in DDPM. This yields to a new equation for $\tilde{\mu}_{\theta} $ expressed in terms of $x_t$ and $\epsilon$:


$$\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}(\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

and hence

$$\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon)$$

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

$$\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon})$$

Hence, the next sample at $t-1$ is sampled from the Gaussian distribution like below:

$$x_{t-1} \sim \mathcal{N}(\tilde{\mu}_\theta,\sigma_t^2 I)$$

...until $x_0$ is reached, in which case only the mean $\tilde{\mu}_\theta$ is extracted as output.

## :bullettrain_front: DDIM: Denoising Diffusion Implicit Model
:construction:

