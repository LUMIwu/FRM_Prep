# Bond Review

## Price
$$
\begin{align}
P &= \sum_{t=1}^{n}{\frac{C_t}{(1+r)^t}} \\
&= \sum_{t=1}^{n}{\frac{C_t}{(1+y)^t}} 
\end{align}
$$

r is yield, y is YTM.

## Duration
#### 1. Macaulay Duration
Macaulay Duration 表示债券未来现金流的加权平均到期时间.

$$
\begin{align}
D_{Macaulay} &= \frac{\sum(\frac{C_t}{(1+y)^t} \times t)}{\sum\frac{C_t}{(1+y)^t}}\\
&= \frac{\sum(\frac{C_t \times t}{(1+y)^t})}{P}
\end{align}
$$

There, $y$ be the YTM.

For **Zero Coupon Bond**, its **Macaulay Duration = Maturity**.

For **Coupon Bond**, its its **Macaulay Duration < Maturity**, 有提前的利息支付分摊了现金流的时间分布。


#### 2. Modified Duration (Sensitivity)
Modified Duration 直接衡量债券价格对利率变化的敏感性。它表示在利率变动 1% 的情况下，债券价格的预期变动百分比.

$$
\begin{align}
D_{mod} &= \frac{D_{Macaulay}}{1+y}\\
&= \frac{\sum(\frac{C_t \times t}{(1+y)^t})}{P(1+y)}\\
&= \frac{1}{P} \times \sum(\frac{C_t \times t}{(1+y)^{t+1}})
\end{align}
$$

如果某债券的 Modified Duration 为 5，说明在利率上升 1% 时，债券价格会下降大约 5%。

- Duration is part of the Derivative, it's **elastisity**.

#### 3. First Derivative of Price to YTM
$$
\begin{align}
\frac{dP}{dy} &= \sum_{t=1}^{n}{\frac{-t \times C_t}{(1+y)^{t+1}}}\\
&= -P \times D_{mod}
\end{align}
$$

这表明，当y变化一个微小单位 $\Delta{yield}$ 时，债券价格P将以 $-P \times D_{mod}$ 的幅度变化。

## Sensitivity
A reasonable approximation of **bond price sensitivity** is given by:

$$
\Delta P \approx \frac{dP}{dy} \times  \Delta y = -D_{mod} \times P \times \Delta y
$$

Where <u>P is the bond price</u> and <u>$\delta y$ is the change in interest rate</u>.

## Convexity
$$
\frac{d^2P}{dy^2} = \sum_{t=1}^{n}{\frac{t \times (t+1) \times C_t}{(1+y)^{t+2}}}
$$

$$
\begin{align}
Convexity &= \frac{1}{P} \times \frac{d^2P}{dy^2} \\&= \frac{1}{P} \times \sum_{t=1}^{n}{\frac{t \times (t+1) \times C_t}{(1+y)^{t+2}}}
\end{align}
$$


## Zero Coupon Bond

### PV
零息债券的价格（现值）是其未来面值的贴现值。

$$
P=\frac{F}{(1+r)^T}
$$

- P: Present Value
- F: Face Value
- r: Interest Rate
- T: Maturity

因为没有定期利息，债券的收益完全由买入价和到期面值之间的差额决定。

### YTM

$$
r = (\frac{F}{P})^{\frac{1}{T}}-1
$$

表示在持有至到期的情况下，投资的年化收益率。

### Duration

**零息债券的Duration等于Maturity(T)**，因为没有中途的现金流。

$$
D = T
$$

Duration是一个衡量Price对Interest Rate Change Sensitivity的指标：

$$
D_{mod} = \frac{T}{1+r}
$$

债券Price的变化与Duration成正比:

$$
\Delta P \approx -D_{mod} \times P \times \Delta r
$$

### Convexity

零息债券的Convexity较高，因为没有中途的现金流。

Convexity是价格和利率非线性关系的调整因子，表示债券价格对利率变化的敏感性变化：

$$
Convexity = \frac{T(T+1)}{(1+r)^2}
$$

**High Convexity**的债券在利率波动时价格变化更平滑。


