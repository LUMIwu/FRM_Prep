## Option Pricing

#### Average Delta

```math
\begin{align}
\bar{\Delta} &= \frac{\Delta_0 + \Delta_1}{2} \\
 &= \frac{\Delta_0 + (\Delta_0 + \gamma * \Delta{S})}{2}
\end{align}
```

#### Taylor Expansion
```math
\begin{align}
∆C \approx ∆*∆S + \frac{1}{2}  \gamma  (∆S)^2 + \nu * ∆\sigma + \Theta * ∆T
\end{align}
```
Where
- ΔC = Change in the call option price
- Δ = Option delta (first derivative of option price with respect to S)
- ΔS = Change in the stock price
- Γ = Option gamma (rate of change of delta)
- Vega 表示期权价格对 <u>波动率变化</u> 的敏感度
- Theta 表示期权价格对 <u>时间衰减</u> 的敏感度

Points: 
1. **Delta (∆)** provides a **linear** approximation of how the option price changes with **S**.
2. **Gamma (Γ)** accounts for the **non-linearity** in the option price change.
3. When ΔS is small, the gamma term is minor, and delta alone provides a good estimate.
4. When **ΔS is large**, **gamma becomes significant**, meaning that delta alone underestimates or overestimates the option price change.

## Derivatives

### Risk-free interest rate

随着 LIBOR 退出，市场更倾向于使用 **SOFR、SONIA、ESTR** 等无风险基准利率。

美元市场：**SOFR** 互换市场快速增长，超过 USD LIBOR 互换市场。
英镑市场：**SONIA** 取代 GBP LIBOR，成为主要的基准利率。
欧元市场：**ESTR** 互换逐渐取代 EURIBOR。
| **无风险利率类型**  | **描述** | **适用场景** |
|------------|---------|---------|
| **美国国债收益率（Treasury Yield）** | 美国政府债券的收益率，传统的无风险利率 | **债券定价、资本成本计算** |
| **SOFR（Secured Overnight Financing Rate）** | 以 **美国国债回购市场（Repo Market）** 为基础的隔夜利率 | **衍生品、贷款、掉期（Swaps）** |
| **SONIA（Sterling Overnight Index Average）** | 以英国银行间市场的隔夜无抵押拆借利率计算 | **英镑市场基准** |
| **ESTR（Euro Short-Term Rate）** | 以欧元区银行拆借市场的隔夜利率计算 | **欧元市场基准** |


### Fixed-to-Floating Contract

Fixed-to-Floating Contract（固定转浮动合约）是一种金融衍生品协议，常用于 利率互换 (Interest Rate Swap)，或债券、贷款等金融工具的结构化安排。

在该合约中：
- 初始阶段支付固定利率（Fixed Rate）
- 随后转换为浮动利率（Floating Rate）
- 浮动利率通常基于市场基准（如 SOFR、SONIA、ESTR）

Application:

#### (1) 利率互换（Interest Rate Swap, IRS）
- **核心作用**：帮助企业或投资者管理利率风险，降低融资成本。
- **基本结构**：
  - 一方支付 **固定利率**（Fixed Leg）
  - 另一方支付 **浮动利率**（Floating Leg）
  - 参考的浮动利率可能是 **SOFR、SONIA、ESTR 或 EURIBOR**
  - 适用于**债券、贷款、资产负债管理**

📌 **示例：**
某公司有 1000 万美元贷款，利率为 **5% 固定利率**。它预计未来利率会下降，因此希望将固定利率转换为浮动利率（基于 SOFR）。  
- 该公司可以与银行签订**固定转浮动利率互换 (Fixed-to-Floating Swap)**
- 结果：公司支付 **SOFR + 1%** 而不再支付 5% 的固定利率。

---

#### (2) 可转换债券（Convertible Bonds）
- **结构**：部分债券可能在**一定时间后从固定利率转换为浮动利率**。
- **好处**：债券发行人可以降低利率成本，而投资者可以获得更高的收益。

📌 **示例：**
某公司发行**5年期可转换债券**：
- **前2年**，持有者获得 **固定利率 4%**
- **第3年起**，转换为 **SOFR + 2%**

这样，投资者可以在市场利率上升时获得更高回报，而公司可以在初期支付较低的固定成本。

---

#### (3) 贷款合约（Fixed-to-Floating Loan）
- **结构**：贷款人在**前几年支付固定利率**，之后切换至浮动利率。
- **用途**：
  - **企业融资**：降低初期融资成本，之后根据市场情况调整支付利率。
  - **个人房贷**：一些房贷产品允许借款人在初期锁定固定利率，之后调整为浮动利率（SOFR + X%）。

📌 **示例：**
银行提供 **10年期房贷**：
- **前 3 年** 固定利率 **3.5%**
- **之后转换为** SOFR + 1.5%

如果 SOFR 低于 2%，那么贷款人最终支付的利率低于 3.5%，从而节省利息成本。

---

#### Pricing 
假设：
- 固定利率 = \( R_f \)
- 浮动利率 = \( R_t \)（如 SOFR + x%）
- 未来的贴现因子 = \( D_t \)
- 现金流 = \( C_t \)
- 总支付价值 \( PV \) 计算方式：

$$
PV = \sum_{t=1}^{N} \frac{C_t}{(1+R_t)^t}
$$

其中：
- 固定利率部分使用固定贴现因子计算
- 浮动利率部分每期根据 SOFR 计算新的贴现值

对于互换交易，通常使用 **无套利定价** 计算固定与浮动现金流的等值点。

## Portfolio

### VaR
| **指标** | **作用** |
|----------|---------|
| **VaR** | 计算整个投资组合在特定置信水平下的最大损失 |
| **iVaR** | 衡量新增或移除资产对 VaR 的影响 |
| **CVaR** | 衡量 VaR 之外的极端损失（尾部风险） |
| **MVaR** | 计算单个资产对 VaR 的贡献度 |

### Expected Shortfall (ES)

### CVaR

### iVaR

iVaR（Incremental Value at Risk，增量风险价值）用于衡量新增或移除某个资产后，对投资组合 **VaR（风险价值）** 的影响。  

$$
iVaR = VaR_{\text{new portfolio}} - VaR_{\text{original portfolio}}
$$

- **iVaR > 0**：新增资产增加组合风险  
- **iVaR < 0**：新增资产降低组合风险（分散化效应）

---

- **完整计算公式**：
$$
\sigma_{\text{new}} = \sqrt{\sigma_P^2 + w_X^2 \sigma_X^2 + 2 w_X \rho_{P,X} \sigma_P \sigma_X}
$$

$$
iVaR = Z_{\alpha} \cdot (\sigma_{\text{new}} - \sigma_P) \cdot \text{Portfolio Value}
$$
- **近似计算公式（适用于小权重资产）**：
$$
iVaR \approx w_X \cdot Z_{\alpha} \cdot \sigma_X \cdot \text{Portfolio Value} \cdot \rho_{P,X}
$$
---
🔹 **iVaR 评估新增资产对 VaR 的影响**，用于投资组合优化和风险管理。  
🔹 **相关性越高，iVaR 越大；负相关资产可降低 VaR**。  
🔹 **适用于银行资本监管、风险对冲、投资组合调整**。  
