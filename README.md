# Probability

## Discrete Distribution
### Binominal Distribution
$$\Large \Large P_r(X = x) = \binom{n}{x}p^x(1-p)^{n-x}$$

$$\Large \mu = np, \ \ \sigma^2 = np(1-p)$$

### Hypergeometric Distribution

$$\Large P_r(X = x) = \frac{\binom{k}{x} \binom{N - k}{n - x}}{\binom{N}{n}}$$

$$\Large \mu =  n\frac{k}{N} ,\ \ \sigma^2 = (\frac{N-n}{N-1})n\frac{k}{N}(1-\frac{k}{N})$$

### Geometric Distribution
$$\Large P_r(X = x) = (1-p)^{x-1}p$$

$$\Large \mu = \frac{1}{p}, \ \ \sigma^2 = \frac{1 - p}{p^2}$$

### Negative Binominal Distribution
$$\Large P_r(X = x) = \binom{x-1}{k-1}p^{k}q^{x-k}$$

$$\Large \mu = \frac{k}{p}, \ \ \sigma^2 = \frac{k(1 - p)}{p^2}$$

### Poisson Distribution
$$\Large P_r(X = x) = \frac{e^{- \lambda t}(\lambda t)^x}{x!}$$

$$\Large \mu = \lambda t, \ \ \sigma^2 =\lambda t$$
#### Binominal Distribution to Poisson Distribution
$$\Large X\sim Poisson(\mu=np)$$

$$\Large \mu = np , \ \ n\rightarrow \infty, p\rightarrow 0$$
## Continuous Distribution
### Uniform Distribution
$$\Large f(x) = \frac{1}{B-A} , A \leq x \leq B$$

$$\Large \mu = \frac{B+A}{2}, \ \ \sigma^2 =\frac{(B-A)^2}{12}$$

### Gaussian Distribution
$$\Large f(x) = \frac{1}{\sqrt{2 \pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$

$$\Large E(x) = \mu, \ \ Var(x) = \sigma^2$$

### Standard Gaussian Distribution
$$\Large X\sim B(\mu, \sigma^2) \rightarrow Z\sim B(0, 1)$$

$$\Large Z = \frac{x-\mu}{\sigma} \rightarrow \Large f(x) = \frac{1}{\sqrt{2 \pi}}e^{-\frac{1}{2}z^2}$$
#### Binominal Distribution to Standard Gaussian Distribution

$$\Large \mu = np, \ \ \sigma^2=npq, \ \ q = (1-p)$$

$$\Large Z = \frac{x - np}{\sqrt{npq}} $$

### Exponential Distribution
$$\Large f_T(t) = \lambda e^{-\lambda t}$$

$$\Large \mu = \frac{1}{\lambda}, \ \ \sigma^2 =\frac{1}{\lambda^2}$$

### Gamma Distribution
$$\Large \Gamma(\alpha) = \int_{0}^{\infty}x^{\alpha - 1}e^{-x}dx, \ \ \alpha > 0 , \ \ \Gamma(1) = 1 , \ \ \Gamma(\frac{1}{2}) = \pi $$

$$\Large f_{T \alpha}(t) = \frac{1}{\beta^{\alpha}\Gamma(\alpha)}t^{\alpha - 1}{e^{\frac{-t}{\beta}}} $$

$$\Large \mu = \alpha \beta, \ \ \sigma^2 =\alpha \beta^2$$

### Chi-square Distribution
$$\Large \alpha = \frac{v}{2}, \ \  \beta = 2$$

$$\Large f(x) = \frac{1}{2^{\frac{v}{2}}\Gamma(\frac{v}{2})}x^{\frac{v}{2} - 1}{e^{\frac{-x}{2}}} $$

$$\Large \mu = v, \ \ \sigma^2 = 2v$$

### Beta Distribution
$$\Large B(\alpha, \beta) = \int_{0}^{1}x^{\alpha - 1}(1-x)^{\beta - 1}dx = \frac{\Gamma{(\alpha})\Gamma{(\beta})}{\Gamma{(\alpha + \beta})}$$

$$\Large f(x) = \frac{1}{B(\alpha, \beta)}x^{\alpha - 1}(1-x)^{\beta - 1}, \ \ 0 < x < 1$$

$$\Large \mu = \frac{\alpha}{\alpha+\beta}, \ \ \sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta + 1)}$$

### Lognormal Distribution
$$\Large f(x) = \frac{1}{\sqrt{2 \pi}\sigma x}e^{-\frac{(ln(x)-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$

$$\Large \mu = e^{\mu + \frac{\sigma^2}{2}}, \ \ \sigma^2 = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$$

### Weibull Distribution
$$\Large f(x) = \alpha \beta x^{\beta - 1}e^{-\alpha x^\beta}$$

$$\Large \mu = \alpha^{-\frac{1}{\beta}}\Gamma(1+\frac{1}{\beta}), \ \ \sigma^2 =  \alpha^{-\frac{2}{\beta}}\{\Gamma(1+\frac{2}{\beta}) - [ \Gamma(1+\frac{1}{\beta})]^2\}$$

### Failure Rate for the Weibull Distribution
$$\Large R(t) = P(T>t) = \int_{t}^{\infty}f(t)dt = 1 - F(t)$$

$$\Large r(t) = \frac{f_T(t)}{1-F_t(t)} = \frac{f_T(t)}{R(t)}$$

$$\Large F_t(t) = 1 - e^{-\int_{0}^{t}r(\tau)d\tau}$$

$$\Large T \sim exp(\lambda)$$

$$\Large f_t(t) = \lambda e^{-\lambda t} , \ \ t>0$$

$$\Large R(t) =\int_{t}^{\infty}\lambda e^{-\lambda t}dt = e^{-\lambda t} $$

$$\Large r(t) = \frac{\lambda e^{-\lambda t}}{ e^{-\lambda t}} = \lambda$$

## Transformations of Variables
### Discrete probability distribution
$$\Large g(y)=f[w(y)]$$
### Discrete joint probability distribution
$$\Large g(y_1,y_2) = f[w_1(y_1, y_2), w_2(y_1, y_2)]$$

### Example:
$$\Large X_1\sim Poisson(\mu_1), \ \ X_2 \sim Poisson(\mu_2), \ \ X_1 \ \ and \ \ X_2 \ \ are \ \ independent$$

$$\Large  Find\ \ the\ \ distribution \rightarrow \ \ Y_1 = X_1 + X2$$
#### Use Transformations of Variables
$$\Large f(x_1,x_2) = f(x_1)f(x_2) = \frac{e^{-\mu_1}\mu_1^{x_1}}{x_1!}\frac{e^{-\mu_2}\mu_2^{x_2}}{x_2!}$$

$$\Large Set \ \ Y_2 = X_2 \rightarrow \begin{Bmatrix}
x_1 = y_1 - y_2\\ 
x_2=y_2 \ \ \ \ \ \ \ \ 
\end{Bmatrix}$$

$$\large f(y_1) = \sum_{y_2=0}^{y_1}\frac{e^{-\mu_1}\mu_1^{(y_1-y_2)}}{(y_1-y_2)!}\frac{e^{-\mu_2}\mu_2^{y_2}}{y_2!}$$

$$\large =e^{-(\mu_1 + \mu_2)}\sum_{y_2=0}^{y_1}\frac{\mu_1^{(y_1-y_2)}\mu_2^{y_2}}{(y_1-y_2)!y_2!}$$

$$\large = \frac{e^{-(\mu_1 + \mu_2)}}{y_1!}\sum_{y_2=0}^{y_1}\binom{y_1}{y_2}\mu_1^{(y_1-y_2)}\mu_2^{y_2}$$

$$\large =  \frac{e^{-(\mu_1 + \mu_2)}(\mu_1 + \mu_2)^{y_1}}{y_1!}$$

$$\large Y_1\sim Poisson(\mu = \mu_1+\mu_2)$$
#### Use Law of Total Probability
$$\large P_r(A) = \sum_{i}P_r(A|B_i)P_r(B_i)$$

$$\large P_r(Y_1 = y) = \sum_{x_2} P_r(Y_1=y_1|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2} P_r(x_1 + x_2=y_1|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2=0}^{y_1} P_r(x_1 =y_1-x_2|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2=0}^{y_1}\frac{e^{-\mu_1}\mu_1^{(y_1-x_2)}}{(y_1-x_2)!}\frac{e^{-\mu_2}\mu_2^{x_2}}{x_2!}$$

$$\large = \frac{e^{-(\mu_1 + \mu_2)}}{y_1!}\sum_{x_2=0}^{y_1}\binom{y_1}{x_2}\mu_1^{(y_1-x_2)}\mu_2^{x_2}$$

$$\large =  \frac{e^{-(\mu_1 + \mu_2)}(\mu_1 + \mu_2)^{y_1}}{y_1!}$$

$$\large Y_1\sim Poisson(\mu = \mu_1+\mu_2)$$

###  Continuous probability distribution
$$\large g(y) = f[w(y)]|J|$$

$$\large J = w'(y) \rightarrow Jacobian$$

### Continuous joint probability distribution
$$\large g(y_1, y_2) = f[w_1(y_1,y_2),w_2(y_1,y_2)]|J|$$

$$\large J = \begin{vmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\ 
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
\end{vmatrix}$$

### Example:

![image](https://hackmd.io/_uploads/H1BJ2Ck_T.png)

$$\large f_x(x)\ \ is\ \ given\ \ -1 < x < 1, \ \ Let \ \ Y = X^2, \ \ f_Y(y) = ?$$

$$\large 1.\rightarrow (-1 < x < 0) ,\ \ f_Y(y) = f_X(-\sqrt{y})|(-\sqrt{y})'|$$

$$\large 2.\rightarrow (0 < x < 1) ,\ \ f_Y(y) = f_X(\sqrt{y})|(\sqrt{y})'|\ \ \ \ $$

$$\large f_Y(y) = \frac{1}{2\sqrt{y}} [f_X(\sqrt{y}) + f_X(-\sqrt{y})]$$

#### Extension
# Probability

## Discrete Distribution
### Binominal Distribution
$$\Large \Large P_r(X = x) = \binom{n}{x}p^x(1-p)^{n-x}$$

$$\Large \mu = np, \ \ \sigma^2 = np(1-p)$$

### Hypergeometric Distribution

$$\Large P_r(X = x) = \frac{\binom{k}{x} \binom{N - k}{n - x}}{\binom{N}{n}}$$

$$\Large \mu =  n\frac{k}{N} ,\ \ \sigma^2 = (\frac{N-n}{N-1})n\frac{k}{N}(1-\frac{k}{N})$$

### Geometric Distribution
$$\Large P_r(X = x) = (1-p)^{x-1}p$$

$$\Large \mu = \frac{1}{p}, \ \ \sigma^2 = \frac{1 - p}{p^2}$$

### Negative Binominal Distribution
$$\Large P_r(X = x) = \binom{x-1}{k-1}p^{k}q^{x-k}$$

$$\Large \mu = \frac{k}{p}, \ \ \sigma^2 = \frac{k(1 - p)}{p^2}$$

### Poisson Distribution
$$\Large P_r(X = x) = \frac{e^{- \lambda t}(\lambda t)^x}{x!}$$

$$\Large \mu = \lambda t, \ \ \sigma^2 =\lambda t$$
#### Binominal Distribution to Poisson Distribution
$$\Large X\sim Poisson(\mu=np)$$

$$\Large \mu = np , \ \ n\rightarrow \infty, p\rightarrow 0$$
## Continuous Distribution
### Uniform Distribution
$$\Large f(x) = \frac{1}{B-A} , A \leq x \leq B$$

$$\Large \mu = \frac{B+A}{2}, \ \ \sigma^2 =\frac{(B-A)^2}{12}$$

### Gaussian Distribution
$$\Large f(x) = \frac{1}{\sqrt{2 \pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$

$$\Large E(x) = \mu, \ \ Var(x) = \sigma^2$$

### Standard Gaussian Distribution
$$\Large X\sim B(\mu, \sigma^2) \rightarrow Z\sim B(0, 1)$$

$$\Large Z = \frac{x-\mu}{\sigma} \rightarrow \Large f(x) = \frac{1}{\sqrt{2 \pi}}e^{-\frac{1}{2}z^2}$$
#### Binominal Distribution to Standard Gaussian Distribution

$$\Large \mu = np, \ \ \sigma^2=npq, \ \ q = (1-p)$$

$$\Large Z = \frac{x - np}{\sqrt{npq}} $$

### Exponential Distribution
$$\Large f_T(t) = \lambda e^{-\lambda t}$$

$$\Large \mu = \frac{1}{\lambda}, \ \ \sigma^2 =\frac{1}{\lambda^2}$$

### Gamma Distribution
$$\Large \Gamma(\alpha) = \int_{0}^{\infty}x^{\alpha - 1}e^{-x}dx, \ \ \alpha > 0 , \ \ \Gamma(1) = 1 , \ \ \Gamma(\frac{1}{2}) = \pi $$

$$\Large f_{T \alpha}(t) = \frac{1}{\beta^{\alpha}\Gamma(\alpha)}t^{\alpha - 1}{e^{\frac{-t}{\beta}}} $$

$$\Large \mu = \alpha \beta, \ \ \sigma^2 =\alpha \beta^2$$

### Chi-square Distribution
$$\Large \alpha = \frac{v}{2}, \ \  \beta = 2$$

$$\Large f(x) = \frac{1}{2^{\frac{v}{2}}\Gamma(\frac{v}{2})}x^{\frac{v}{2} - 1}{e^{\frac{-x}{2}}} $$

$$\Large \mu = v, \ \ \sigma^2 = 2v$$

### Beta Distribution
$$\Large B(\alpha, \beta) = \int_{0}^{1}x^{\alpha - 1}(1-x)^{\beta - 1}dx = \frac{\Gamma{(\alpha})\Gamma{(\beta})}{\Gamma{(\alpha + \beta})}$$

$$\Large f(x) = \frac{1}{B(\alpha, \beta)}x^{\alpha - 1}(1-x)^{\beta - 1}, \ \ 0 < x < 1$$

$$\Large \mu = \frac{\alpha}{\alpha+\beta}, \ \ \sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta + 1)}$$

### Lognormal Distribution
$$\Large f(x) = \frac{1}{\sqrt{2 \pi}\sigma x}e^{-\frac{(ln(x)-\mu)^2}{2\sigma^2}}, -\infty \leq x \leq \infty$$

$$\Large \mu = e^{\mu + \frac{\sigma^2}{2}}, \ \ \sigma^2 = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$$

### Weibull Distribution
$$\Large f(x) = \alpha \beta x^{\beta - 1}e^{-\alpha x^\beta}$$

$$\Large \mu = \alpha^{-\frac{1}{\beta}}\Gamma(1+\frac{1}{\beta}), \ \ \sigma^2 =  \alpha^{-\frac{2}{\beta}}\{\Gamma(1+\frac{2}{\beta}) - [ \Gamma(1+\frac{1}{\beta})]^2\}$$

### Failure Rate for the Weibull Distribution
$$\Large R(t) = P(T>t) = \int_{t}^{\infty}f(t)dt = 1 - F(t)$$

$$\Large r(t) = \frac{f_T(t)}{1-F_t(t)} = \frac{f_T(t)}{R(t)}$$

$$\Large F_t(t) = 1 - e^{-\int_{0}^{t}r(\tau)d\tau}$$

$$\Large T \sim exp(\lambda)$$

$$\Large f_t(t) = \lambda e^{-\lambda t} , \ \ t>0$$

$$\Large R(t) =\int_{t}^{\infty}\lambda e^{-\lambda t}dt = e^{-\lambda t} $$

$$\Large r(t) = \frac{\lambda e^{-\lambda t}}{ e^{-\lambda t}} = \lambda$$

## Transformations of Variables
### Discrete probability distribution
$$\Large g(y)=f[w(y)]$$
### Discrete joint probability distribution
$$\Large g(y_1,y_2) = f[w_1(y_1, y_2), w_2(y_1, y_2)]$$

### Example:
$$\Large X_1\sim Poisson(\mu_1), \ \ X_2 \sim Poisson(\mu_2), \ \ X_1 \ \ and \ \ X_2 \ \ are \ \ independent$$

$$\Large  Find\ \ the\ \ distribution \rightarrow \ \ Y_1 = X_1 + X2$$
#### Use Transformations of Variables
$$\Large f(x_1,x_2) = f(x_1)f(x_2) = \frac{e^{-\mu_1}\mu_1^{x_1}}{x_1!}\frac{e^{-\mu_2}\mu_2^{x_2}}{x_2!}$$

$$\Large Set \ \ Y_2 = X_2 \rightarrow \begin{Bmatrix}
x_1 = y_1 - y_2\\ 
x_2=y_2 \ \ \ \ \ \ \ \ 
\end{Bmatrix}$$

$$\large f(y_1) = \sum_{y_2=0}^{y_1}\frac{e^{-\mu_1}\mu_1^{(y_1-y_2)}}{(y_1-y_2)!}\frac{e^{-\mu_2}\mu_2^{y_2}}{y_2!}$$

$$\large =e^{-(\mu_1 + \mu_2)}\sum_{y_2=0}^{y_1}\frac{\mu_1^{(y_1-y_2)}\mu_2^{y_2}}{(y_1-y_2)!y_2!}$$

$$\large = \frac{e^{-(\mu_1 + \mu_2)}}{y_1!}\sum_{y_2=0}^{y_1}\binom{y_1}{y_2}\mu_1^{(y_1-y_2)}\mu_2^{y_2}$$

$$\large =  \frac{e^{-(\mu_1 + \mu_2)}(\mu_1 + \mu_2)^{y_1}}{y_1!}$$

$$\large Y_1\sim Poisson(\mu = \mu_1+\mu_2)$$
#### Use Law of Total Probability
$$\large P_r(A) = \sum_{i}P_r(A|B_i)P_r(B_i)$$

$$\large P_r(Y_1 = y) = \sum_{x_2} P_r(Y_1=y_1|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2} P_r(x_1 + x_2=y_1|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2=0}^{y_1} P_r(x_1 =y_1-x_2|X_2=x_2)P_r(X_2=x_2)$$

$$\large = \sum_{x_2=0}^{y_1}\frac{e^{-\mu_1}\mu_1^{(y_1-x_2)}}{(y_1-x_2)!}\frac{e^{-\mu_2}\mu_2^{x_2}}{x_2!}$$

$$\large = \frac{e^{-(\mu_1 + \mu_2)}}{y_1!}\sum_{x_2=0}^{y_1}\binom{y_1}{x_2}\mu_1^{(y_1-x_2)}\mu_2^{x_2}$$

$$\large =  \frac{e^{-(\mu_1 + \mu_2)}(\mu_1 + \mu_2)^{y_1}}{y_1!}$$

$$\large Y_1\sim Poisson(\mu = \mu_1+\mu_2)$$

###  Continuous probability distribution
$$\large g(y) = f[w(y)]|J|$$

$$\large J = w'(y) \rightarrow Jacobian$$

### Continuous joint probability distribution
$$\large g(y_1, y_2) = f[w_1(y_1,y_2),w_2(y_1,y_2)]|J|$$

$$\large J = \begin{vmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\ 
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
\end{vmatrix}$$

### Example:
<div style="text-align:center;">
<img src="https://hackmd.io/_uploads/H1BJ2Ck_T.png" alt="image" />
</div>
$$\large f_x(x)\ \ is\ \ given\ \ -1 < x < 1, \ \ Let \ \ Y = X^2, \ \ f_Y(y) = ?$$

$$\large 1.\rightarrow (-1 < x < 0) ,\ \ f_Y(y) = f_X(-\sqrt{y})|(-\sqrt{y})'|$$

$$\large 2.\rightarrow (0 < x < 1) ,\ \ f_Y(y) = f_X(\sqrt{y})|(\sqrt{y})'|\ \ \ \ $$

$$\large f_Y(y) = \frac{1}{2\sqrt{y}} [f_X(\sqrt{y}) + f_X(-\sqrt{y})]$$

#### Extension
<div style="text-align:center;">
    <img src="https://hackmd.io/_uploads/rk8KAAyup.png" alt="image" />
</div>


$$\large f_x(x)\ \ is\ \ given\ \ -1 < x < 2, \ \ Let \ \ Y = X^2, \ \ f_Y(y) = ?$$

$$
\large f_Y(y) = \left\{
\begin{matrix}
\frac{1}{2\sqrt{y}} [f_X(\sqrt{y}) + f_X(-\sqrt{y})] & \text{if} \ 0 < y < 1 \\
\frac{1}{2\sqrt{y}} f_X(\sqrt{y}) & \text{if} \ 1 < y < 4
\end{matrix}
\right.
$$

## Moment-Generating Functions
$$\large \mu_r' = E(X^r) = \left\{
\begin{matrix}
\sum_{x} X^r f(x) \ \ \ \ & \text{if} \ X \ is\ discrete  \\
\int_{-\infty}^{\infty} X^r f(x)dx  & \text{if}\ X \ is\ continuous
\end{matrix}
\right.
$$

$$\large f_x(x)\ \ is\ \ given\ \ -1 < x < 2, \ \ Let \ \ Y = X^2, \ \ f_Y(y) = ?$$

$$
\large f_Y(y) = \left\{
\begin{matrix}
\frac{1}{2\sqrt{y}} [f_X(\sqrt{y}) + f_X(-\sqrt{y})] & \text{if} \ 0 < y < 1 \\
\frac{1}{2\sqrt{y}} f_X(\sqrt{y}) & \text{if} \ 1 < y < 4
\end{matrix}
\right.
$$

## Moment-Generating Functions
$$\large \mu_r' = E(X^r) = \left\{
\begin{matrix}
\sum_{x} X^r f(x) \ \ \ \ & \text{if} \ X \ is\ discrete  \\
\int_{-\infty}^{\infty} X^r f(x)dx  & \text{if}\ X \ is\ continuous
\end{matrix}
\right.
$$
