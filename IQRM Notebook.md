# IQRM Notebook
# Chapter 1 - basic math

## Logarithm
$$
\begin{align}
log_b a &= x ,\, then \: a = b^x \nonumber\\

log_b b^x &= x \nonumber\\

log_b XY &= log_b X + log_b Y \nonumber\\

log_b kX &= log_b k + log_b X \nonumber\\
\end{align}
$$

## Log Returns

1. Simple returns: $R_t = \frac{P_t-P_{t-1}}{P_{t-1}}$
2. Log returns: $r_t = ln{(1+R_t)}$
3. Approx of log returns: $ r \approx R -\frac{1}{2} R^2 $

### Cumulative Compounding Returns
$$
\begin{align}

R_{n,t} &= \frac{P_t-P_{t-n}}{P_{t-n}} = \frac{P_t}{P_{t-n}}-1 \nonumber\\

&= \frac{P_t}{P_{t-1}}\frac{P_{t-1}}{P_{t-2}}\frac{P_{t-2}}{P_{t-3}}...\frac{P_{t-n+1}}{P_{t-n}}-1 \nonumber\\

R_{n,t} &= (1+R_{1,t})(1+R_{1,t-1})...(1+R_{1,t-n+1})-1 \nonumber\\

1+R_{n,t} &= (1+R_{1,t})(1+R_{1,t-1})...(1+R_{1,t-n+1}) \nonumber\\

&使 \nonumber \\

r_{n,t} &= ln(1+R_{n,t})\nonumber\\
&两边取对数 \nonumber \\
r_{n,t} &= r_{1,t}+r_{1,t-1}+...+r_{1,t-n+1} \nonumber
\end{align}
$$
### Limited Liability

Simple return: [-100%, +$\infty$]

Log return: [-$\infty$, +$\infty$] , which is easier to calculate

## Elasticity
$$
Elasticity=\frac{\%Change In Y}{\%Change In X}=\frac{∆y/y}{∆x/x} = \frac{dy}{dx}\times\frac{x}{y}
$$

# Chapter 3 - Stats

## Samples

### Khintchine's Law of Large Numbers 大数定律

If you do the same random experiment over and over again, the average outcome will approach the true expected value (or mean) as the number of trials gets larger.

### Simple Random Sampling

$$
X_1, \cdots, X_n \stackrel{i.i.d.}{\sim} X
$$

- i.i.d.: independent and identically distributed

If we assume the population has a cumulative distribution function (CDF) $F$, then the joint distribution function of the n samples could be denoted by $F_n(x_1,x_2,\cdots, x_n)$

$$
F_n(x_1,x_2,\cdots, x_n) = F(x_1)F(x_2) \cdots F(x_n)
$$

Same, for pdf $f$, we have:

$$
f_n(x_1,x_2,cdots, x_n) = f(x_1)f(x_2) \cdots f(x_n)
$$

These are sample distributions.

### Sample Mean

$$
\bar{X}=\frac{1}{n} \sum_{j=1}^n X_j
$$

### Sample Variance

$$
S^2=\frac{1}{n-1} \sum_{j=1}^n\left(X_j-\bar{X}\right)^2
$$

- $S^2$ is an unbiased estimator of the true population variance $\sigma^2$

### Order Statistics 次序统计量

$$
X_{(1)} \leq X_{(2)} \leq \ldots \leq X_{(n)}
$$

- Mean: $X_{(1)}$
- Max: $X_{(n)}$
- Medium: $X_{(\frac{n+1}{2})}$ if n is odd
- Quantiles: $X_{(\frac{n}{4})}$ for the 25th percentile.
- PDF for $k$th Order Statistics:

$$
f_{X_{(k)}}(x)=\frac{n!}{(k-1)!(n-k)!}[F(x)]^{k-1}[1-F(x)]^{n-k} f(x)
$$

### Sample Moments 样本矩

Raw Moments about the origin

$$
m_k=\frac{1}{n} \sum_{i=1}^n X_i^k
$$

k-th Central Moments

$$
\mu_k=\frac{1}{n} \sum_{i=1}^n (X_i-\bar{X})^k
$$

- First Moment (Mean): $m_k=\frac{1}{n} \sum_{i=1}^n X_i^k$
- ​Second Central Moment (Variance): $\mu_k=\frac{1}{n} \sum_{i=1}^n (X_i-\bar{X})^k$
- ​Third Central Moment (Skewness)​: $Skewness=\frac{\mu_3}{\mu_2^{3/2}}$
- ​Fourth Central Moment (Kurtosis): $Kurtosis = \frac{\mu_4}{\mu_2^2}$

# Chapter 4 - Distributions

## Bernoulli Trial

$$
p(X_i=1) = p \quad p(X_i=0) = 1-p
$$

$$
E[X]=\sum{x_ip}=1*p+0*(1-p)=p
$$

$$
V(X) = E[X-E[X]]^2 = (1-p)^2*p +(0-p)^2*(1-p) = p(1-p)
$$

## Binomial Distribution

Probability Mass Function:

$$
f(k,n,p)=Pr(X=k) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

$$
where\quad \binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

mean & variance derive from bernoulli trial, Since each $X_i$ is a Bernoulli random variable

$$
E[X] = E[X_1+X_2+...+X_n] = E[X_1]+E[X_2]+...+E[X_n] = n*p
$$

$$
Var(X) = Var(X_1+X_2+...+X_n) = Var(X_1) + Var(X_2) + ... + Var(X_n) = n*p*(1-p)
$$

## Poisson Distribution

derive from Binomial Distribution

As n → $\infin$ and p → 0, with the mean $\lambda = n*p$ kept constant, the binomial distribution approaches the **Poisson distribution** Poisson(λ), whose PMF is:

$$
P(X=k) = \frac{\lambda^ke^{-\lambda}}{k!}
$$

## Normal Distribution

## Student's T Distribution

1. Mean: The mean of the t-distribution is 0 (same as Norm Dist)
   
   $$
   E(T) = 0
   $$
2. Variance: The variance of a t-distribution depends on the degrees of freedom k.
   
   $$
   Var(T) = \frac{k}{k-2} \quad for \quad k >2
   $$
3. Hypothesis Test
   
   - Step1: ​H0: The population mean $\mu = \mu_0$.
   - Step2: Calculate the T-statistic.
     
     $$
     t=\frac{\bar{X}-\mu_0}{\frac{s}{\sqrt{n}}}
     $$
   - Step3: Determine the Critical Value.
   
   $$
   t_c = T.INV.2T(Probability,df)
   $$
   
   - Step4: Compare. If |t| > $t_c$, then reject H0.

![student's-t-distribution - CFA, FRM, and Actuarial Exams Study Notes](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDC2mJP4CyAPqw8_jKD7nqEq-XFTjuNAdL3w&s)

## F Distribution

## Chi-Square Distribution

$$
\frac{(n-1) S^2}{\sigma^2} \sim \chi_{n-1}^2
$$

follows a chi-square distribution with (n-1) degrees of freedom

- **Mean**: The mean of a chi-square distribution with $k$ degrees of freedom is $k$.
  
  $$
  \mathbb{E}\left[\frac{(n-1) S^2}{\sigma^2}\right]=n-1
  $$
  
  rearrange it,
  
  $$
  \mathbb{E}(S^2)=\sigma^2
  $$
- **Variance**: The variance of a chi-square distribution is 2$k$.
  
  $$
  Var(\frac{(n-1) S^2}{\sigma^2})=2(n-1)
  $$
  
  rearrange it,
  
  $$
  Var(S^2) = \frac{2 \sigma^4}{n-1}
  $$
- **Hypothesis Test**
  Step1: H0: The two variables are independent
  
  Step2: calculate Chi-Square Test Statistics
  
  $$
  \chi ^2 = \frac{df*Observed Var}{Expected Var}
  $$
  
  Step3: Calculate the critical value. (P=0.05 when confidence level = 95%)
  
  $$
  \chi_c^2 = CHISQ.INV.RT(Probability, df)
  $$
  
  Step4: Compare. If $\chi ^2 > \chi_c^2$, then reject H0.

![Understanding Chi-Square Critical Value: A Beginner's Tutorial](https://www.easysevens.com/wp-content/uploads/2024/01/Chi-Squared-Distribution.png)


## mixture distributions

$ f(x) = \sum_{i=1}^n \omega_i f_i(x) \quad \text{subject to} \quad \sum_{i=1}^n \omega_i = 1 $

Let's say adding two normal distributions together, we can create many distributions.

(1) Biomodal Mixture Distribution

(2) Skewed Mixture Distribution

(3) Exceed Kurtosis Mixture Distribution

# Chapter 5 - Multivariate Distributions and Copulas

# Chapter 7 - Hypothesis Testing

## sample mean as random variable

### Mean of sample mean

$$
\hat{µ} = \frac{1}{n}\sum{x_i} = \sum{\frac{1}{n}x_i}
$$

### Variance of sample mean

$$
\sigma_{\hat{µ}}^2 = \frac{\sigma^2}{n}
$$

with its derivation:

$$
\begin{aligned}
 \sigma_{\hat{µ}}^2 = V(\hat{µ}) &= V(\frac{1}{n}\sum{x_i})\\
&= (\frac{1}{n})^2V(\sum{x_i})\\
&= (\frac{1}{n})^2(V(x_1)+V(x_2)+...+V(x_n) \\
&= (\frac{1}{n})^2 \times n \times \sigma^2 = \frac{\sigma^2}{n}

\end{aligned}
$$

### Variance of the sample Variance

$$
Var(S^2) = \frac{2 \sigma^4}{n-1}
$$

## sample variance as random variable

$$
MSE = \frac{1}{n}\sum{(Y_i-\hat{Y}_i)^2}
$$

$$
MSE(\hat{\theta}) = E[(\hat{\theta}-\theta)^2]
$$

where
$\hat{\theta}$: the estimator (e.g., sample variance, sample mean, etc.).
$\theta$: true population parameter (e.g., population variance, population mean)..

## Chebyshev's Inequality

For a random variable with mean $\mu$ and variance $\sigma^2$,

$$
\mathbb{P}(|X-mu| \geq k \sigma) \leq \frac{1}{k^2}
$$

- $\mathbb{P}$ is the probability that $X$ is at least $k$ std away from the mean.(No more than $\frac{1}{k^2}$ of the data can be more than $k$ std away from the mean.)
- $k$ is any positive number greater than 1.
- if k = 2, then at most 1/4 = 25% of the data points can be more than 2 std away from the mean.

# Chapter 9 - Financa Model

## Correlation

Corr(X,Y) = Cov(X, Y) / Sqrt(Var(X)*Var(Y))

## Covariance

Cov(X, Y) = E[(X-E[X)(Y-E[Y])] = E[XY] – E[X]E[Y]

## 

## Sharp Ratio

$$
S_a = \frac{E[R_a-R_b)]}{\sigma_a}
$$
Where $R_a$ is asset return, $R_b$ is risk-free return, $\sigma_a$ is standard deviation of the asset excess return.

## Optimal Hedge Ratio
$$
n=\frac{Cov(S,F)}{Var(F)}
$$

## Portfolio Variance

$$
Var(R_p) = w^T S w
$$

Where $w$ is a $n \times 1$ column vector of asset weights,

$S$ is a $n \times n$ **Covariance matrix** of asset returns.

$S_{ij}$ is the Covariance between the returns of asset i and asset j

The diagonal elements $S_{ii}$ are the **variances of individual assets**.

The portfolio variance can also be written as:

$$
Var(R_p) = \sum_{i=1}^n \sum_{j=1}^n w_i w_j S_{ij}
$$

**If all assets are independent:**

The covariance matrix S is diagonal, and the portfolio variance simplifies to:

$$
Var(R_p) = \sum_{i=1}^n w_i^2 \sigma_i^2
$$

where $\sigma_i^2$ is the variance of asset i.