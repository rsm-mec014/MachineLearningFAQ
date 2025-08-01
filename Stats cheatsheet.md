# 🧠 Machine Learning & Regression Analysis Interview Cheat Sheet

## ✈️ Topics Covered
- XGBoost
- Linear Regression
- Statistical Inference
- Regression Theory (SE, Variance, Degrees of Freedom)
- VIF and Model Diagnostics

---

## 🔶 1. XGBoost (Extreme Gradient Boosting)

### Core Idea
- Ensemble of decision trees using **gradient boosting**.
- Learns **residuals** of the previous trees.
- Optimizes:  
  **Objective = Training Loss + Regularization (L1 + L2)**

### Key Components
- **Shrinkage** (learning rate): Slows down updates.
- **Subsampling**: Rows/columns for regularization.
- **Missing value handling**: Auto-learn split direction.
- **Second-order Taylor expansion** for optimization.

### Loss Function (Binary Classification)
\[
\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
\]

### Overfitting Control
- `max_depth`, `min_child_weight`
- `subsample`, `colsample_bytree`
- `lambda` (L2), `alpha` (L1)

---

## 📉 2. Linear Regression

### Model
\[
Y = X\beta + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I)
\]

### OLS Estimator
\[
\hat{\beta} = (X^TX)^{-1} X^TY
\]

### Variance of Estimator
\[
\text{Var}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}
\]

### Residual Variance Estimate
\[
\hat{\sigma}^2 = \frac{1}{n - p} \sum (y_i - \hat{y}_i)^2
\]

### Standard Error of \( \hat{\beta}_j \)
\[
SE(\hat{\beta}_j) = \sqrt{ \hat{\sigma}^2 \cdot [(X^TX)^{-1}]_{jj} }
\]

### R-squared & Adjusted R-squared
\[
R^2 = 1 - \frac{RSS}{TSS}, \quad \text{Adj } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
\]

---

## 📊 3. Statistical Inference

### Hypothesis Testing
- t-test: compare means or coefficients
- z-test: for proportions or known variance
- F-test: for overall model fit
- Chi-square test: categorical data

### Confidence Intervals
- For sample mean:
\[
\bar{x} \pm z^* \cdot \frac{\sigma}{\sqrt{n}}
\]
- For coefficients:
\[
\hat{\beta}_j \pm t^* \cdot SE(\hat{\beta}_j)
\]

### Notes
- p-value tells **significance**, CI shows **plausible range**
- Watch for Type I/II errors
- Avoid p-hacking or peeking

---

## 🔎 4. SE vs Variance vs Std. Deviation

| Term | Definition | Formula | Purpose |
|------|------------|---------|---------|
| Variance | Spread of data | \( \text{Var}(X) = \mathbb{E}[(X - \mu)^2] \) | Descriptive |
| Std. Deviation | Square root of variance | \( \sigma = \sqrt{\text{Var}} \) | Descriptive |
| Standard Error | SD of an estimator | \( SE(\hat{\theta}) = \sqrt{ \text{Var}(\hat{\theta}) } \) | Inference |

### Common SEs

| Estimator | SE Formula |
|-----------|------------|
| Sample mean \( \bar{X} \) | \( \frac{s}{\sqrt{n}} \) |
| Sample proportion \( \hat{p} \) | \( \sqrt{ \frac{p(1 - p)}{n} } \) |
| OLS coefficient \( \hat{\beta}_j \) | See above |
| Difference in means | \( \sqrt{ \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2} } \) |

---

## 📐 5. Degrees of Freedom (df)

| Context | Degrees of Freedom |
|---------|--------------------|
| Sample variance | \( n - 1 \) |
| Regression residuals | \( n - p \) or \( n - p - 1 \) |
| t-test (1 sample) | \( n - 1 \) |
| t-test (2 samples) | \( n_1 + n_2 - 2 \) |
| F-test | \( (p, n - p - 1) \) |
| VIF regression | \( n - k - 1 \) |

### Mnemonics
- **Sample variance**: lose 1 for estimating mean → \( n - 1 \)
- **Regression**: lose \( p+1 \) for coefficients → \( n - p - 1 \)
- **F-test**: df = model vs. residual

---

## 📊 6. VIF (Variance Inflation Factor)

### Purpose:
Measures multicollinearity by how much the variance of \( \hat{\beta}_j \) is inflated.

\[
VIF_j = \frac{1}{1 - R_j^2}
\]

Where:
- \( R_j^2 \) is from regressing \( X_j \) on all other predictors.

### Intuition:
- If \( X_j \) can be well predicted by others → \( R_j^2 \) high → VIF high
- \( \text{VIF} > 5 \sim 10 \) = strong multicollinearity

---

## ✅ 7. Common Regression Interview Questions

| Question | Short Answer |
|----------|--------------|
| How to check model quality? | R², F-test, residual plots |
| What if residuals are not normal? | Try transformation or non-parametric models |
| What if predictors are correlated? | Use VIF, Ridge regression, or drop features |
| Difference between SE and std dev? | SE measures **estimation uncertainty**; std dev is data variability |
| Why use OLS? | It's BLUE: Best Linear Unbiased Estimator under Gauss-Markov assumptions |

---

## 🧳 8. STAR Framework for Regression Interview Answers

1. **Assumptions**: Is the model applicable?
2. **Estimation**: How are coefficients estimated?
3. **Inference**: Are estimates statistically significant?
4. **Interpretation**: What do coefficients mean?
5. **Diagnostics**: Is the model valid?
6. **Improvement**: How can it be improved?

---

## 📚 Reference Mnemonics

- **SE ≠ std dev**: SE is for estimates
- **df = n - p - 1**: in regression
- **VIF > 10 = warning sign**
- **R² always ↑ with features → use adjusted R²**

---

