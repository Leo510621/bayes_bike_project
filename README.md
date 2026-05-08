# Bayesian Bike Sharing Project — Module D

Focus on extension task C: Sequential Bayesian Updating.

**Module D (Ke)** is responsible for:
- Extension Task C: Sequential Bayesian Updating across 4 chronological chunks
- Chunk diagnostic statistics
- Report sections: §6 Extension Task, §7 Conclusions
- Report integration

## Reproduction

```bash
pip install -r requirements.txt

# Step 1: Run Sequential Bayesian Updating
#         (writes plots and table)
python -m src.sequential_update

# Step 2: Generate chunk diagnostics
#         (writes chunk diagnosis table)
python -m src.chunk_diagnostics

# Step 3: Run tests
pytest tests/
```

### ⚡ Performance Note (Windows)

By default, PyTensor runs in pure Python mode, which is slow.
Installing a C++ compiler reduces this to ~1 min/chunk.

**Option A — Rtools (recommended for Windows):**
1. Download Rtools44 from https://cran.r-project.org/bin/windows/Rtools/
2. Install to any drive (e.g. `D:\rtools44`)
3. In each new terminal session, add g++ to PATH:
```powershell
$env:PATH += ";D:\rtools44\x86_64-w64-mingw32.static.posix\bin"
```
4. Verify: `g++ --version`
5. Run `python -m src.sequential_update`

**Option B — conda:**
```bash
conda install -c conda-forge m2w64-toolchain
```

## Cross-Module Contract

### Input from Module A

`data/day_subsampled.csv` (360 rows, `random_state=42`), with columns:

```
dteday, cnt, temp, hum, season, yr, mnth, holiday, weekday, workingday,
weathersit, atemp, windspeed, casual, registered, instant
```
Only `cnt`, `temp`, `hum` are used in the model.
Slicing convention: `df.iloc[i*90:(i+1)*90]` for chunk i (0-indexed).

### Input from Module B

Prior hyperparameters inherited as starting point for Chunk 1:

```
beta_0, beta_1, beta_2 ~ Normal(0, 10000^2)
sigma                  ~ HalfNormal(scale = 3000)
```

### Random seed

All Module D code uses `random_seed = 42 + chunk_idx` one per chunk, for full reproducibility.

## Model

Sequential Updating cascades the posterior of chunk k as the prior for chunk k+1
via moment matching:

```
Chunk 1 prior:  beta_j ~ Normal(0, 10000^2),  sigma ~ HalfNormal(3000)
                                    ↓
Chunk 2 prior:  beta_j ~ Normal(mu_1^j, sd_1^j),  sigma ~ HalfNormal(mu_1^sigma)
                                    ↓
Chunk 3 prior:  beta_j ~ Normal(mu_2^j, sd_2^j),  sigma ~ HalfNormal(mu_2^sigma)
                                    ↓
Chunk 4 prior:  beta_j ~ Normal(mu_3^j, sd_3^j),  sigma ~ HalfNormal(mu_3^sigma)
```

Each chunk uses the same NUTS configuration as Module B:
4 chains × 2000 draws (+ 1000 tune steps discarded as burn-in) = 8000 retained posterior samples per chunk.

## Project Structure

```
bayes_bike_project-Module-D/
├── data/
│   └── day_subsampled.csv                    # from Module A
├── src/
│   ├── __init__.py
│   ├── config.py                             # paths, constants, MCMC hyperparameters
│   ├── sequential_update.py                  # Sequential Updating, cascade plot
│   └── chunk_diagnostics.py                  # chunk descriptive stats
├── results/
│   ├── sequential_posterior_table.csv        
│   ├── chunk_diagnostics.csv                 
│   └── figures/
│       ├── cascade_plot.png                  
│       └── density_evolution.png             
├── report/
│   ├── main.tex                              
│   ├── section6_extension.tex               
│   └── section7_conclusions.tex             
├── tests/
│   └── test_sequential_update.py
├── requirements.txt
├── .gitignore
├── pyproject.toml
└── README.md
```
