# Bayesian Bike Sharing

### Bayesian linear regression for daily bike rentals, fitted with NUTS in PyMC 5 on the UCI Bike Sharing dataset.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5](https://img.shields.io/badge/PyMC-5.x-red.svg)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![QMUL](https://img.shields.io/badge/Queen%20Mary-QHM6703-005EB8.svg)](https://www.qmul.ac.uk/)

---

## Overview

We model daily bike rentals as a linear function of normalised temperature and humidity. The likelihood is Gaussian, the priors are weakly informative, and we sample the posterior of $(\beta_0, \beta_1, \beta_2, \sigma)$ with NUTS.

The analysis has four parts. An OLS fit gives a point-estimate sanity check. Convergence diagnostics confirm that the four chains mix. A sensitivity check compares weakly informative, very diffuse, and aggressively narrow priors. A sequential update splits the data into four chronological blocks and passes the posterior forward as the next block's prior.

Dataset: UCI [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) ([Fanaee-T & Gama, 2014](https://doi.org/10.1007/s13748-013-0040-3)), $n = 360$ subsample with seed 42, sorted by date.

| Variable | Mean | SD | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| `cnt` (daily rentals) | 4392.11 | 1935.64 | 22 | 4336.5 | 8555 |
| `temp` (normalised) | 0.486 | 0.182 | 0.097 | 0.476 | 0.849 |
| `hum` (normalised) | 0.615 | 0.145 | 0.000 | 0.608 | 0.973 |

### Headline results

A 47 °C swing across the year (the full range of normalised temperature) raises expected daily rentals by about **+6800**. A move from a perfectly dry to a saturated atmosphere lowers them by about **−2350**. Residual scale is **σ ≈ 1500**, around 34% of the mean — calendar effects, holidays, and other weather covariates account for the rest. All four chains converged ($\hat R \le 1.002$, ESSbulk ≥ 3 775, zero divergences).

---

## What's worth flagging

A few choices in this analysis are a bit less standard, and worth pointing out:

- **The Poisson likelihood is rejected at the conditional level, not the marginal level.** A var/mean ratio of 853 looks damning, but most of that comes from across-day variation in the conditional mean (winter vs summer), not from overdispersion within a day. The argument for Gaussian rests on the CLT and the distance from the zero boundary.

- **Prior hyperparameters are anchored to the empirical scale of `cnt`.** The $\beta$-priors have SD 10,000, covering roughly seven times the observed range. The Half-Normal(3000) prior on $\sigma$ has a median near 2024, comparable to the sample SD of 1936. Neither number is pulled out of thin air.

- **The Bayesian and OLS results agree to two decimal places, but for a specific reason.** With these priors and $n = 360$, the data dominates the posterior. Section 5 stress-tests this: a tight Prior C breaks the agreement immediately.

- **Sequential updating exposes a sign flip in the humidity coefficient.** The full-sample posterior puts $\beta_2$ at $-2350$, but block-level posteriors flip from negative in winter to positive in spring/summer. Within high-temperature blocks, humidity tracks temperature locally, and the partial effect of humidity inverts.

- **$\sigma$ rises monotonically across the four sequential blocks (652 → 1503).** This independently corroborates the Breusch–Pagan rejection of homoscedasticity in the OLS fit, and motivates a heteroscedastic extension for future work.

---

## Pipeline

```
Raw day.csv (731 obs)
        │
        ▼
┌──────────────────┐
│  data_prep.py    │   seed=42, n=360, sorted by date           ┐
└──────────────────┘                                            │
        │                                                       │  Module-A
        ▼                                                       │  (this branch)
┌──────────────────┐                                            │
│  eda.py          │   summary stats, scatter, correlation      │
└──────────────────┘                                            │
        │                                                       │
        ▼                                                       │
┌──────────────────────┐                                        │
│ frequentist_baseline │   OLS reference fit                    ┘
└──────────────────────┘
        │
        ▼
┌──────────────────────┐                                        ┐
│ model_pymc           │   NUTS, 4 chains × 2000                │
│  + posterior_summary │   warmup 1000, target 0.95             │  module-B
└──────────────────────┘                                        ┘
        │
        ▼
┌──────────────────────┐                                        ┐
│ diagnostics.py       │   trace, R̂, rank, ESS, autocorr        │
└──────────────────────┘                                        │  module-c
        │                                                       │
        ▼                                                       │
┌──────────────────────┐                                        │
│ sensitivity.py       │   Priors A / B / C                     ┘
└──────────────────────┘
        │
        ▼
┌──────────────────────┐                                        ┐
│ sequential_update.py │   4 blocks × moment-matched prior      │  Module-D
└──────────────────────┘                                        ┘
```

Each downstream module lives on its own branch and consumes the artefacts produced upstream.

---

## Model specification

$$
\begin{aligned}
\beta_0,\beta_1,\beta_2 &\overset{\text{iid}}{\sim} \mathcal{N}(0,\ 10000^2)\\
\sigma &\sim \mathrm{Half\text{-}Normal}(3000)\\
\mu_i &= \beta_0 + \beta_1\,\mathrm{temp}_i + \beta_2\,\mathrm{hum}_i\\
\mathrm{cnt}_i \mid \mu_i, \sigma &\sim \mathcal{N}(\mu_i,\ \sigma^2),\quad i = 1,\dots,360
\end{aligned}
$$

The 95% prior interval $(-19{,}600,\ 19{,}600)$ for each $\beta_j$ easily covers the OLS estimates ($\hat\beta_1 \approx 6796$, $\hat\beta_2 \approx -2370$), and excludes only values that would imply an effect larger than the full range of `cnt`. The Half-Normal scale of 3000 puts the prior median at about 2024, close to the empirical SD of `cnt` (1936).

### Prior predictive check

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/prior_predictive.png)

500 prior predictive draws span roughly ±30,000, about seven times the empirical range of `cnt`. The mass is centred at zero, so the prior does not pre-determine the sign or scale of the response.

---

## Exploratory data analysis

| | |
| :-: | :-: |
| ![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/cnt_histogram.png) | ![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/correlation_heatmap.png) |
| ![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/cnt_vs_temp_scatter.png) | ![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/cnt_vs_hum_scatter.png) |

The marginal correlation between temperature and `cnt` is $r = 0.612$. Humidity's marginal correlation is only $-0.079$, which understates its conditional effect: temperature and humidity have a modest positive correlation in this subsample ($r = 0.155$), so part of humidity's negative influence on `cnt` gets absorbed into temperature in the marginal view. The posterior recovers a credibly negative partial effect once temperature is conditioned on.

---

## Posterior inference

We sample with NUTS in PyMC 5: 4 chains, 2000 retained draws after 1000 warm-up steps, target acceptance 0.95, seed 42. No divergent transitions were recorded.

| Parameter | Mean | SD | 95% HDI lower | 95% HDI upper | $\hat R$ | ESSbulk |
| --- | --- | --- | --- | --- | --- | --- |
| $\beta_0$ (intercept) | 2541 | 382 | 1799 | 3324 | 1.00 | 3 775 |
| $\beta_1$ (temp) | **6784** | 436 | 5915 | 7632 | 1.00 | 5 371 |
| $\beta_2$ (hum) | **−2350** | 555 | −3457 | −1276 | 1.00 | 4 181 |
| $\sigma$ | 1500 | 56 | 1393 | 1610 | 1.00 | 5 352 |

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/posterior_density.png)

The 95% HDI for $\beta_1$ and $\beta_2$ both lie entirely on one side of zero.

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/posterior_forest.png)

### Interpreting the coefficients

On the physical scale:

- **Temperature.** The normalised range $[0, 1]$ corresponds to a 47 °C swing across the year. The posterior implies this swing raises expected daily rentals by about +6800.
- **Humidity.** Going from perfectly dry to saturated lowers expected rentals by about 2350 on average.
- **Intercept.** $\beta_0 \approx 2541$ corresponds to a near-freezing, perfectly arid day, a combination that does not occur in the data. The intercept is not directly interpretable here.
- **Residual scale.** $\sigma \approx 1500$ is about 34% of the mean of `cnt`. Calendar effects, holidays, and the other weather covariates we leave out account for the rest of the variation.

---

## Convergence diagnostics

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/trace_plots.png)

All four chains show stable trajectories with no drift, and overlap across chains throughout the post-warm-up region.

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/autocorr.png)

Autocorrelation decays to zero within 5–10 lags.

| Diagnostic | Result |
| --- | --- |
| $\hat R$ | ≤ 1.002 across all parameters (threshold 1.01) |
| ESSbulk | 3 775 to 5 371 (rule of thumb: ≥ 400) |
| Divergent transitions | 0 / 8000 |
| Burn-in | 1000 warm-up steps; chains stationary within 100–200 post-warm-up draws |

---

## Frequentist comparison

The OLS fit and the Bayesian posterior produce essentially identical point estimates and intervals.

| Parameter | OLS point | Bayes mean | OLS 95% CI | Bayes 95% HDI |
| --- | --- | --- | --- | --- |
| $\beta_0$ | 2547.58 | 2541.46 | [1801.30, 3293.87] | [1799.45, 3323.63] |
| $\beta_1$ | 6795.86 | 6783.75 | [5931.57, 7660.15] | [5914.93, 7632.40] |
| $\beta_2$ | −2370.23 | −2350.28 | [−3452.04, −1288.42] | [−3457.29, −1276.42] |
| $\sigma$ | 1497.40 | 1500.30 | — | [1393, 1610] |

This numerical agreement does not make the two approaches interchangeable. The frequentist 95% CI is a frequency property of the procedure that produced the interval. The Bayesian HDI is a probability statement about the parameter given the data. They coincide here because the priors carry almost no information and $n = 360$ is enough for the likelihood to dominate. Section 5's Prior C breaks the agreement.

---

## Prior sensitivity

We compare three priors:

|  | $\beta_j$ | $\sigma$ |
| --- | --- | --- |
| **Prior A** (baseline) | $\mathcal{N}(0,\ 10000^2)$ | Half-Normal(3000) |
| **Prior B** (very diffuse) | $\mathcal{N}(0,\ 100000^2)$ | Half-Normal(30000) |
| **Prior C** (narrow stress test) | $\mathcal{N}(0,\ 1000^2)$ | Half-Normal(300) |

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/sensitivity_comparison.png)

A and B produce essentially the same posteriors. The largest standardised shift $\Delta / \mathrm{SD}_\text{post}$ between them is 0.05, which is well within sampling noise. Increasing the prior scale by a factor of ten does nothing.

Prior C is a different story. $\hat\beta_1$ shifts toward zero by 2.17 posterior SDs, $\hat\beta_2$ by 1.78 SDs. CI widths contract by 10–20%. If you pick a prior tight enough to disagree with the likelihood, the prior wins. This confirms that the agreement with OLS in Section 3.2 follows from the weakly informative prior, not from anything that holds automatically.

---

## Extension: sequential Bayesian updating

We sort the 360 observations by date and split them into four blocks of 90, one per half-year across 2011–2012. Each block fits the same model. The posterior from one block becomes the prior for the next via moment matching: we match the mean and SD of each marginal, keeping $\beta_j$ as Normal and $\sigma$ as Half-Normal. Joint posterior correlations between parameters are dropped.

$$
\beta_j \mid \text{Block }k+1 \sim \mathcal{N}\!\left(\mu_k^{(j)},\ (s_k^{(j)})^2\right), \qquad
\sigma \mid \text{Block }k+1 \sim \mathrm{Half\text{-}Normal}\!\left(\frac{\mu_k^{(\sigma)}}{\sqrt{2/\pi}}\right)
$$

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/cascade_plot.png)

![](https://github.com/Leo510621/bayes_bike_project/raw/Module-A/results/figures/density_evolution.png)

### What sequential updating reveals

- **Temperature ($\beta_1$).** Standard Bayesian learning. The posterior SD shrinks by 40% across the four blocks. The posterior mean stays positive throughout.

- **Humidity ($\beta_2$).** The posterior mean migrates from $-313$ in Block 1 through near-zero in Blocks 2–3 to $+456$ in Block 4. In high-demand summer/autumn blocks, humidity covaries locally with temperature, and the partial effect of humidity within those blocks turns positive. The full-sample posterior averages this seasonal pattern out into a credibly negative net effect, and is the more reliable estimate.

- **Residual scale $\sigma$.** Rises from 652 in Block 1 to 1503 in Block 4. Low-demand winter days fluctuate less in absolute terms than high-demand summer days. This is the same heteroscedasticity that the Breusch–Pagan test flagged in the OLS diagnostics ($p < 10^{-3}$), now appearing along a different axis.

The takeaway is that moment-matched sequential updating works for strong, stable signals like $\beta_1$ and breaks down on weak ones like $\beta_2$. Dropping the joint correlations between $(\beta_1, \beta_2, \sigma)$ means each block's prior loses information that the full joint posterior would have kept. Strong signals can survive that loss. Weak ones get eaten by it.

---

## Reproducing the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Download the UCI dataset (skips if already present)
python -m src.download_data

# 2. Subsample to n = 360 with seed 42, sorted by date
python -m src.data_prep

# 3. Exploratory plots and summary statistics
python -m src.eda

# 4. OLS baseline
python -m src.frequentist_baseline

# 5. Tests
pytest tests/
```

The Bayesian pipeline (NUTS sampling, diagnostics, sensitivity, sequential updating) lives on the sibling branches `module-B`, `module-c`, and `Module-D`. Each branch is self-contained for its module.

### Cross-module data contracts

`day_subsampled.csv` has a fixed column order: `dteday, cnt, temp, hum, season, yr, mnth, holiday, weekday, workingday, weathersit, atemp, windspeed, casual, registered, instant`. Rows are sorted chronologically. Sequential updating slices `df.iloc[i*90:(i+1)*90]`.

`frequentist_results.csv` has schema `parameter, mle, std_err, ci_lower, ci_upper, p_value`, with `parameter ∈ {beta_0, beta_1_temp, beta_2_hum, sigma}`. Module B's `posterior_table.csv` uses the same parameter names so the two tables can be joined directly.

Every module uses `random_state=42` / `np.random.seed(42)`.

---

## Repository layout

```
bayes_bike_project/
├── data/
│   ├── day.csv                       # Raw UCI data (auto-downloaded)
│   └── day_subsampled.csv            # n = 360, seed = 42
├── src/
│   ├── config.py                     # Paths, contracts, logging
│   ├── data_prep.py                  # Load, validate, subsample
│   ├── download_data.py              # UCI dataset fetch with fallback URL
│   ├── eda.py                        # Summary stats + plots
│   └── frequentist_baseline.py       # OLS reference fit
├── results/
│   ├── figures/                      # All plots used in this README
│   ├── eda_summary_stats.csv
│   ├── frequentist_results.csv
│   ├── frequentist_diagnostics.csv
│   └── overdispersion_check.txt
├── report/                           # LaTeX sources for §1, §2.1, §3.2
├── tests/                            # pytest unit tests
├── requirements.txt
└── README.md
```

---

## Team

| Member | Responsibilities |
| --- | --- |
| **Yuelin Wu** | §1 Introduction and dataset, §2.1 Likelihood justification, §3.2 Frequentist comparison |
| **Bowen Jiang** | §2.2 Prior specification, §3.1 Bayesian posterior summaries |
| **Zhe Han** | §4 Sampler behaviour, §5 Prior sensitivity |
| **Keming Chen** | §6 Sequential updating extension, §7 Conclusions, report integration |
| **Letian Xu** | Dataset selection, cross-section support |

QHM6703 Bayesian Statistics, Queen Mary University of London. Hainan, May 2026.

---

## References

1. Fanaee-T, H. and Gama, J. (2014). [Event labeling combining ensemble detectors and background knowledge](https://doi.org/10.1007/s13748-013-0040-3). *Progress in Artificial Intelligence*, 2(2–3), 113–127.
2. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall/CRC.
3. Gelman, A. and Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457–472.
4. Hoffman, M. D. and Gelman, A. (2014). [The No-U-Turn Sampler](https://jmlr.org/papers/v15/hoffman14a.html). *JMLR*, 15(47), 1593–1623.
5. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.-C. (2021). [Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC](https://doi.org/10.1214/20-BA1221). *Bayesian Analysis*, 16(2), 667–718.
6. Abril-Pla, O. et al. (2023). [PyMC: a modern, comprehensive probabilistic programming framework in Python](https://doi.org/10.7717/peerj-cs.1516). *PeerJ Computer Science*, 9, e1516.
