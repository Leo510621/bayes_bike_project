"""
Core Task 1: Bayesian linear regression via PyMC.
Model (standardised space): cnt_z ~ N(β0 + β1·temp_z + β2·hum_z, σ²)
All variables z-score standardised so β are O(1); σ=10 is genuinely weak.
Saves posterior trace to results/trace.nc
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from src.config import DATA_DIR, RESULTS_DIR, SUBSAMPLED_FILE, RANDOM_SEED

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / SUBSAMPLED_FILE)

# z-score standardise
cnt_mean, cnt_std   = df["cnt"].mean(),  df["cnt"].std()
temp_mean, temp_std = df["temp"].mean(), df["temp"].std()
hum_mean,  hum_std  = df["hum"].mean(),  df["hum"].std()

cnt_z  = ((df["cnt"]  - cnt_mean)  / cnt_std).values
temp_z = ((df["temp"] - temp_mean) / temp_std).values
hum_z  = ((df["hum"]  - hum_mean)  / hum_std).values

# Save scaling constants for back-transformation
import json
scales = dict(cnt_mean=cnt_mean, cnt_std=cnt_std,
              temp_mean=temp_mean, temp_std=temp_std,
              hum_mean=hum_mean, hum_std=hum_std)
(RESULTS_DIR / "scaling.json").write_text(json.dumps(scales, indent=2))

with pm.Model():
    beta0 = pm.Normal("beta0", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=10)
    beta2 = pm.Normal("beta2", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu    = beta0 + beta1 * temp_z + beta2 * hum_z
    pm.Normal("cnt_obs", mu=mu, sigma=sigma, observed=cnt_z)
    idata = pm.sample(draws=2000, tune=1000, chains=4,
                      random_seed=RANDOM_SEED, target_accept=0.9,
                      progressbar=True)

az.to_netcdf(idata, RESULTS_DIR / "trace.nc")
print(f"Saved: {RESULTS_DIR / 'trace.nc'}")
