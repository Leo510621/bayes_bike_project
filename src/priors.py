"""
priors.py — Weakly Informative Prior Definitions

Specify prior distributions for the 4 unknown parameters in the Bayesian linear regression:
    cnt_i ~ N(beta0 + beta1 * temp_i + beta2 * hum_i, sigma^2)

Design Principle (QHM §3.2): Weakly Informative Priors
  - Exclude clearly implausible values
  - Do not strongly favor any specific value
  - Do not constrain the posterior; let the data dominate

Hyperparameter Selection (Source: Yue's results/eda_summary_stats.csv):
  - SD(cnt) ≈ 1936, predictors temp/hum ∈ [0, 1]
  - OLS estimates: beta0 ≈ 2548, beta1 ≈ 6796, beta2 ≈ -2370, sigma_hat ≈ 1497
  - True parameters are on the order of thousands, so prior dispersion needs to be 10^4 to be "weak"
"""

from __future__ import annotations

import pymc as pm


# Prior Hyperparameters

# Priors for regression coefficients beta0, beta1, beta2: N(mean=0, sd=10000)
#   - mean = 0: No prior assumption on the direction of the effect (let data determine sign)
#   - sd   = 10000: 95% prior interval ±20000, covering all plausible values
#                   Excludes absurd values (|beta| > 5e4 is impossible at the data scale)
#                   This prior is effectively flat relative to the true beta (thousands scale)
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 10000.0

# Prior for residual standard deviation sigma: HalfNormal(scale=3000)
#   - Must be > 0 (standard deviation cannot be negative), automatically enforced by HalfNormal
#   - scale = 3000: prior median ≈ 2024, 95th percentile ≈ 5880
#   - Same order of magnitude as SD(cnt) ≈ 1936, covers OLS residual sigma_hat = 1497
#   - Excludes implausible values of sigma > 10000
SIGMA_PRIOR_SCALE = 3000.0


# Prior Definition Functions

def define_priors() -> dict:
    """
    Create all prior random variables within a PyMC model context.

    Must be called inside a `with pm.Model() as model:` block.

    Returns
    -------
    dict
        Parameter name -> PyMC random variable.
        Keys: "beta0", "beta1", "beta2", "sigma"
    """
    beta0 = pm.Normal("beta0", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    beta1 = pm.Normal("beta1", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    beta2 = pm.Normal("beta2", mu=BETA_PRIOR_MEAN, sigma=BETA_PRIOR_SD)
    sigma = pm.HalfNormal("sigma", sigma=SIGMA_PRIOR_SCALE)
    return {"beta0": beta0, "beta1": beta1, "beta2": beta2, "sigma": sigma}


def prior_summary() -> str:
    """
    Return a human-readable summary of the prior specifications
    (ready for direct copy into report §2.2).
    """
    return (
        f"beta0 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"beta1 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"beta2 ~ Normal(mu={BETA_PRIOR_MEAN}, sigma={BETA_PRIOR_SD})\n"
        f"sigma ~ HalfNormal(sigma={SIGMA_PRIOR_SCALE})"
    )


if __name__ == "__main__":
    # Print specifications to the terminal for debugging
    print(prior_summary())
