# Assignment 5 — Statistical Modeling and Cosmological Parameter Estimation

## Overview

### Exercise 01 — Unbiased Variance Estimator

* Prove that
  $$\hat{\sigma}^2 = \frac{{\text{RSS}}(\hat{\beta})}{N - p}$$
  is an **unbiased estimator** of the true variance $\sigma^2$ under the linear model
  $Y = X\beta + \varepsilon$,
  where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ are i.i.d. noise terms.

### Exercise 02 — Generalized Least Squares (GLS)

* Starting from the definition
  $$\text{RSS}_\Sigma(\beta) = e^\top \Sigma^{-1} e$$
  show that the **maximum likelihood estimator** of $\beta$ is given by
  $$\hat{\beta} = (X^\top \Sigma^{-1} X)^{-1} X^\top \Sigma^{-1} Y.$$
* This demonstrates how correlated observational errors modify the classical OLS estimator.

### Exercise 03 — Cross-Entropy Loss from Boltzmann Distribution

* Derive the **cross-entropy loss** function
  $$L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{c} y_{n,i} \log(\hat{y}_{n,i})$$
  from the Boltzmann distribution assumption.
* Discuss its probabilistic interpretation and connection to maximum likelihood estimation for classification tasks.

### Exercise 04 — Cosmological Parameter Estimation

* Two-phase workflow to study cosmological parameters using the **matter power spectrum** $P(k)$:

  * **Phase 1 — Physics Intuition:** Use **CAMB** to generate $P(k)$ for various values of $\Omega_m$ and $H_0$, and analyze the physical trends.
  * **Phase 2 — Machine Learning Regression:** Train regression models to predict cosmological parameters from the simulated power spectra.
* Evaluate model performance and interpret how $P(k)$ encodes cosmological information.

