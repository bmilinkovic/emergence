# src/my_toolbox/models/var.py

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

class VarModel:
    """
    A wrapper class around statsmodels VAR functionality.
    """

    def __init__(self, maxlags=15, criterion='aic'):
        """
        Initialize the VAR model.

        Parameters
        ----------
        maxlags : int, optional
            The maximum number of lags to consider for model order selection.
        criterion : str, optional
            The criterion for order selection. One of {'aic', 'bic', 'hqic'}.
        """
        if criterion not in ['aic', 'bic', 'hqic']:
            raise ValueError("criterion must be one of 'aic', 'bic', or 'hqic'")
        self.maxlags = maxlags
        self.criterion = criterion
        self.model = None
        self.results = None
        self.selected_order = None

    def fit(self, data: pd.DataFrame):
        """
        Fit a VAR model to the provided data after selecting the best model order.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame where each column is one time series variable.

        Returns
        -------
        results : VARResults
            The fitted VAR model results.

        Raises
        ------
        ValueError
            If data is not a DataFrame or has insufficient data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.shape[0] < 2:
            raise ValueError("Not enough observations to fit a VAR model.")

        self.model = VAR(data)
        order_results = self.model.select_order(maxlags=self.maxlags)
        # Select best order based on chosen criterion
        if self.criterion == 'aic':
            self.selected_order = order_results.aic
        elif self.criterion == 'bic':
            self.selected_order = order_results.bic
        else:
            self.selected_order = order_results.hqic

        # Fallback if order selection fails
        if self.selected_order is None or np.isnan(self.selected_order):
            self.selected_order = 1

        self.results = self.model.fit(self.selected_order)
        return self.results

    def compute_spectral_radius(self):
        """
        Compute the spectral radius of the fitted VAR model.

        The spectral radius is the maximum absolute eigenvalue of the companion matrix of the VAR.

        Returns
        -------
        radius : float
            The spectral radius.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if self.results is None:
            raise RuntimeError("Model must be fitted before computing spectral radius.")

        # For a VAR(p) model with k variables, the companion matrix is (kp x kp):
        # [A1     A2     ...    Ap-1    Ap
        #  I      0      ...    0       0
        #  0      I      ...    0       0
        #  ...                          ...
        #  0      0      ...    I       0]
        #
        # A1, A2, ... Ap are the coefficient matrices of shape (k, k).
        # We'll construct this matrix and find its eigenvalues.

        coefs = self.results.coefs  # shape (p, k, k)
        p, k, _ = coefs.shape

        # Build companion matrix
        companion = np.zeros((k*p, k*p))
        # Fill top row blocks
        for i in range(p):
            companion[0:k, i*k:(i+1)*k] = coefs[i]

        # Fill lower block with identity matrices
        if p > 1:
            companion[k:(p*k), 0:(k*(p-1))] = np.eye(k*(p-1))

        eigenvalues = np.linalg.eigvals(companion)
        radius = max(abs(eigenvalues))
        return radius

    def print_model_info(self):
        """
        Print model info: selected model order, spectral radius, and a summary of the fitted VAR model.
        """
        if self.results is None:
            print("Model not fitted yet.")
            return
        print(f"Selected Order: {self.selected_order}")
        radius = self.compute_spectral_radius()
        print(f"Spectral Radius: {radius:.4f}")
        print(self.results.summary())

    def forecast(self, steps=1):
        """
        Forecast future values using the fitted VAR model.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        forecast : np.ndarray
            Array of forecasted values with shape (steps, n_vars).
        """
        if self.results is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        return self.results.forecast(self.results.y, steps)


def simulate_var(coef_matrix: np.ndarray, intercept: np.ndarray, n_obs: int, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_vars = coef_matrix.shape[0]
    data = np.zeros((n_obs, n_vars))
    eps = np.random.randn(n_obs, n_vars)

    for t in range(1, n_obs):
        data[t] = intercept + data[t-1] @ coef_matrix.T + eps[t]

    return pd.DataFrame(data, columns=[f"var{i}" for i in range(n_vars)])

