"""
test_model_pymc.py — Fast Unit Tests for Module B Model Code

Does NOT run full MCMC (takes minutes), only validates:
- Prior hyperparameters match expected values (prevent accidental changes)
- define_priors correctly creates 4 variables in a PyMC context
- build_model successfully constructs a model object with synthetic data
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from src.priors import (
    BETA_PRIOR_MEAN,
    BETA_PRIOR_SD,
    SIGMA_PRIOR_SCALE,
    define_priors,
)


# Hard-coded checks for prior hyperparameters

def test_prior_hyperparameters() -> None:
    """Prior hyperparameters must match the values declared in Report §2.2."""
    assert BETA_PRIOR_MEAN == 0.0
    assert BETA_PRIOR_SD == 10000.0
    assert SIGMA_PRIOR_SCALE == 3000.0


# Behavior of define_priors

def test_define_priors_returns_four_variables() -> None:
    """The prior function must return exactly 4 variables."""
    with pm.Model():
        priors = define_priors()
    assert set(priors.keys()) == {"beta0", "beta1", "beta2", "sigma"}


# Behavior of build_model

@pytest.fixture
def fake_data():
    """Generate synthetic data for testing model construction."""
    rng = np.random.default_rng(0)
    n = 100
    return {
        "cnt":  rng.uniform(0, 9000, n),
        "temp": rng.uniform(0, 1, n),
        "hum":  rng.uniform(0, 1, n),
    }


def test_build_model_returns_pymc_model(fake_data) -> None:
    """build_model must return a valid pm.Model instance containing 4 unobserved random variables."""
    from src.model_pymc import build_model

    model = build_model(fake_data["cnt"], fake_data["temp"], fake_data["hum"])
    assert isinstance(model, pm.Model)

    free_var_names = {rv.name for rv in model.free_RVs}
    assert {"beta0", "beta1", "beta2", "sigma"} <= free_var_names
