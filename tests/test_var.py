import pytest
import numpy as np
import pandas as pd
from emergence.models.var import VarModel, simulate_var

def test_var_model_stability():
    # Coefficients chosen to ensure stability
    # Diagonal entries are less than 1 in magnitude
    A = np.array([[0.3, 0.0],
                  [0.0, 0.3]])
    intercept = np.array([0.1, -0.1])
    data = simulate_var(A, intercept, n_obs=500, seed=42)

    model = VarModel(maxlags=5, criterion='aic')
    results = model.fit(data)

    assert results is not None, "VAR model fitting returned None."

    # Ensure a valid order is chosen (or fallback works)
    assert isinstance(model.selected_order, int) and model.selected_order > 0, f"No valid order selected: {model.selected_order}"

    radius = model.compute_spectral_radius()
    # Check that the spectral radius is less than 1, indicating stability
    assert radius < 1.0, f"Spectral radius {radius} is not less than 1, model might be unstable."

    forecast = model.forecast(steps=5)
    assert forecast.shape == (5, data.shape[1]), "Forecast shape is incorrect."

