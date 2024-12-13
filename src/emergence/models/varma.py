# src/my_toolbox/models/varma.py

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

class VarmaModel:
    """
    A wrapper class around statsmodels VARMAX functionality to fit VARMA models.
    """

    def __init__(self, order=(1,1)):
        """
        Initialize the VARMA model.

        Parameters
        ----------
        order : tuple (p, q)
            The (p, q) order of the VARMA model.
        """
        self.order = order
        self.model = None
        self.results = None

    def fit(self, data: pd.DataFrame):
        """
        Fit a VARMA model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame where each column is one time series variable.

        Raises
        ------
        ValueError
            If data is not a DataFrame or is too small.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[0] <= max(self.order):
            raise ValueError("Not enough observations to fit a VARMA model given the specified order.")

        self.model = VARMAX(data, order=self.order, trend='c')
        self.results = self.model.fit(disp=False)
        return self.results

    def forecast(self, steps=1):
        """
        Forecast future values using the fitted VARMA model.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        forecast : pd.DataFrame
            DataFrame of forecasted values.
        """
        if self.results is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        return self.results.forecast(steps=steps)

    def summary(self):
        """
        Print the summary of the fitted VARMA model.
        """
        if self.results is not None:
            return self.results.summary()
        else:
            return "Model not fitted yet."


def simulate_varma(phi: list, theta: list, intercept: np.ndarray, n_obs: int, seed=None):
    """
    Simulate a VARMA(p,q) process (simplified for demonstration).

    The VARMA(p,q) model can be written as:
    X_t = intercept + sum_{i=1}^p Phi_i * X_{t-i} + sum_{j=1}^q Theta_j * eps_{t-j} + eps_t

    This function simulates data from a specified VARMA model given the coefficient 
    matrices and intercept. For simplicity, assume p=1, q=1 (VARMA(1,1)):

    X_t = intercept + Phi_1 * X_{t-1} + Theta_1 * eps_{t-1} + eps_t

    Parameters
    ----------
    phi : np.ndarray
        The Phi_1 matrix of shape (n_vars, n_vars).
    theta : np.ndarray
        The Theta_1 matrix of shape (n_vars, n_vars).
    intercept : np.ndarray
        Intercept vector of shape (n_vars,).
    n_obs : int
        Number of observations to simulate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : pd.DataFrame
        Simulated data with shape (n_obs, n_vars).
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = phi.shape[0]
    data = np.zeros((n_obs, n_vars))

    eps = np.random.randn(n_obs, n_vars)

    for t in range(1, n_obs):
        # For VARMA(1,1):
        # X_t = intercept + Phi_1 X_{t-1} + Theta_1 eps_{t-1} + eps_t
        data[t] = intercept
        data[t] += data[t-1] @ phi.T
        data[t] += eps[t-1] @ theta.T
        data[t] += eps[t]

    return pd.DataFrame(data, columns=[f"var{i}" for i in range(n_vars)])

