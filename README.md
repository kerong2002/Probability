# Probability
<style>
.left-align {
    text-align: left;
}
</style>

<span class="left-align">$$P_r(X = x) = \binom{n}{x}p^x(1-p)^{n-x}$$</span>

## Discrete Distribution
### Binominal Distribution

$$P_r(X = x) = \binom{n}{x}p^x(1-p)^{n-x}$$
$$\mu = np, \ \ \sigma^2 = np(1-p)$$

### Hypergeometric Distribution

$$P_r(X = x) = \frac{\binom{k}{x} \binom{N - k}{n - x}}{\binom{N}{n}}$$
$$\mu =  n\frac{k}{N} ,\ \ \sigma^2 = (\frac{N-n}{N-1})n\frac{k}{N}(1-\frac{k}{N})$$

### Geometric Distribution
$$P_r(X = x) = (1-p)^{x-1}p$$
$$\mu = \frac{1}{p}, \ \ \sigma^2 = \frac{1 - p}{p^2}$$

### Negative Binominal Distribution
$$P_r(X = x) = \binom{x-1}{k-1}p^{k}q^{x-k}$$
$$\mu = \frac{k}{p}, \ \ \sigma^2 = \frac{k(1 - p)}{p^2}$$

### Poisson Distribution
$$P_r(X = x) = \frac{e^{- \lambda t}(\lambda t)^x}{x!}$$
$$\mu = \lambda t, \ \ \sigma^2 =\lambda t$$
#### Binominal Distribution to Poisson Distribution
$$X\sim Poisson(\mu=np)$$
$$\mu = np , \ \ n\rightarrow \infty, p\rightarrow 0$$
## Continuous Distribution
### Uniform Distribution
$$f(x) = \frac{1}{B-A} , A \leq x \leq B$$
$$\mu = \frac{B+A}{2}, \ \ \sigma^2 =\frac{(B-A)^2}{12}$$

### Gaussian Distribution
$$f(x) = \frac{1}{\sqrt{2 \pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$
$$E(x) = \mu, \ \ Var(x) = \sigma^2$$

### Standard Gaussian Distribution
$$X\sim B(\mu, \sigma^2) \rightarrow Z\sim B(0, 1)$$
$$Z = \frac{x-\mu}{\sigma} $$
#### Binominal Distribution to Standard Gaussian Distribution
$$\mu = np, \ \ \sigma^2=npq, \ \ q = (1-p)$$
$$ Z = \frac{x - np}{\sqrt{npq}} $$

### Exponential Distribution
$$f_T(t) = \lambda e^{-\lambda t}$$
$$\mu = \frac{1}{\lambda}, \ \ \sigma^2 =\frac{1}{\lambda^2}$$

### Gamma Distribution
$$ \Gamma(\alpha) = \int_{0}^{\infty}x^{\alpha - 1}e^{-x}dx, \ \ \alpha > 0 , \ \ \Gamma(1) = 1 , \ \ \Gamma(\frac{1}{2}) = \pi $$

$$f_{T \alpha}(t) = \frac{1}{\beta^{\alpha}\Gamma(\alpha)}t^{\alpha - 1}{e^{\frac{-t}{\beta}}} $$
$$\mu = \alpha \beta, \ \ \sigma^2 =\alpha \beta^2$$

### Chi-square Distribution
$$ \alpha = \frac{v}{2}, \ \  \beta = 2$$
$$f(x) = \frac{1}{2^{\frac{v}{2}}\Gamma(\frac{v}{2})}x^{\frac{v}{2} - 1}{e^{\frac{-x}{2}}} $$
$$\mu = v, \ \ \sigma^2 = 2v$$

### Beta Distribution
$$B(\alpha, \beta) = \int_{0}^{1}x^{\alpha - 1}(1-x)^{\beta - 1}dx = \frac{\Gamma{(\alpha})\Gamma{(\beta})}{\Gamma{(\alpha + \beta})}$$
$$f(x) = \frac{1}{B(\alpha, \beta)}x^{\alpha - 1}(1-x)^{\beta - 1}, \ \ 0 < x < 1$$
$$\mu = \frac{\alpha}{\alpha+\beta}, \ \ \sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta + 1)}$$

### Lognormal Distribution
$$f(x) = \frac{1}{\sqrt{2 \pi}\sigma x}e^{-\frac{(ln(x)-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$
$$\mu = e^{\mu + \frac{\sigma^2}{2}}, \ \ \sigma^2 = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$$

