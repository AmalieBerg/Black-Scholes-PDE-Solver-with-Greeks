"""
Greeks calculation: analytical (Black-Scholes) and numerical (PDE grid).
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class GreeksResult:
    """Container for all Greek values."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    method: str  # 'analytical' or 'numerical'


class AnalyticalGreeks:
    """Black-Scholes analytical Greeks for validation."""
    
    @staticmethod
    def calculate(
        S0: float = None,  # Allow both S and S0
        K: float = None,
        r: float = None,
        q: float = None,
        sigma: float = None,
        T: float = None,
        option_type: str = 'call',
        S: float = None,  # Legacy compatibility
        **kwargs
    ) -> GreeksResult:
        """
        Calculate analytical Greeks using Black-Scholes formulas.
        
        Parameters same as PDE solver.
        
        Returns:
        --------
        GreeksResult with all Greeks
        """
        # Handle both S and S0 parameter names
        if S0 is None and S is None:
            raise ValueError("Must provide either S0 or S")
        stock_price = S0 if S0 is not None else S
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            return GreeksResult(0, 0, 0, 0, 0, 'analytical')
        
        # Black-Scholes d1, d2
        d1 = (np.log(stock_price / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Common terms
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # Standard normal PDF
        
        if option_type.lower() == 'call':
            # Call Greeks
            delta = np.exp(-q * T) * N_d1
            theta = (
                -stock_price * n_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * N_d2
                + q * stock_price * np.exp(-q * T) * N_d1
            )
            rho = K * T * np.exp(-r * T) * N_d2
        else:  # put
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = (
                -stock_price * n_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * stock_price * np.exp(-q * T) * norm.cdf(-d1)
            )
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Greeks independent of option type
        gamma = n_d1 * np.exp(-q * T) / (stock_price * sigma * np.sqrt(T))
        vega = stock_price * np.exp(-q * T) * n_d1 * np.sqrt(T)
        
        # Convert theta to per-day (from per-year)
        theta_daily = theta / 365.0
        
        # Convert vega to per 1% change (from per 100%)
        vega_percent = vega / 100.0
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega_percent,
            theta=theta_daily,
            rho=rho / 100.0,  # Per 1% rate change
            method='analytical'
        )


class NumericalGreeks:
    """Greeks calculated from PDE grid."""
    
    @staticmethod
    def calculate_from_grid(
        grid: np.ndarray,
        stock_prices: np.ndarray,
        S0: float,
        dt: float
    ) -> dict:
        """
        Calculate Greeks from PDE grid at t=0.
        
        Parameters:
        -----------
        grid : np.ndarray
            PDE solution grid (time x stock)
        stock_prices : np.ndarray
            Stock price grid points
        S0 : float
            Current stock price
        dt : float
            Time step size
            
        Returns:
        --------
        dict with delta, gamma, theta at S0
        """
        # Find index closest to S0
        idx = np.argmin(np.abs(stock_prices - S0))
        
        # Ensure we're not at boundaries
        if idx == 0:
            idx = 1
        elif idx == len(stock_prices) - 1:
            idx = len(stock_prices) - 2
        
        # Delta: ∂V/∂S using central difference
        dS = stock_prices[idx + 1] - stock_prices[idx - 1]
        dV = grid[0, idx + 1] - grid[0, idx - 1]
        delta = dV / dS
        
        # Gamma: ∂²V/∂S²
        V_plus = grid[0, idx + 1]
        V_mid = grid[0, idx]
        V_minus = grid[0, idx - 1]
        dS_plus = stock_prices[idx + 1] - stock_prices[idx]
        dS_minus = stock_prices[idx] - stock_prices[idx - 1]
        
        gamma = 2 * (V_plus / (dS_plus * (dS_plus + dS_minus))
                     - V_mid / (dS_plus * dS_minus)
                     + V_minus / (dS_minus * (dS_plus + dS_minus)))
        
        # Theta: ∂V/∂t (using forward difference in time)
        if grid.shape[0] > 1:
            theta = (grid[1, idx] - grid[0, idx]) / dt
            theta_daily = theta / 365.0
        else:
            theta_daily = 0.0
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily
        }
    
    @staticmethod
    def calculate_vega_rho(
        solver_class,
        base_price: float,
        **solver_params
    ) -> dict:
        """
        Calculate Vega and Rho using bump-and-revalue.
        
        Parameters:
        -----------
        solver_class : class
            PDE solver class
        base_price : float
            Base option price
        solver_params : dict
            Parameters for solver
            
        Returns:
        --------
        dict with vega and rho
        """
        # Vega: bump sigma by 1% (absolute)
        params_vega = solver_params.copy()
        sigma_bump = 0.01  # 1% absolute vol change
        params_vega['sigma'] = params_vega['sigma'] + sigma_bump
        solver_vega = solver_class(**params_vega)
        result_vega = solver_vega.solve(N_time=100, N_stock=200)
        vega = result_vega.price - base_price  # Per 1% vol change
        
        # Rho: bump r by 1% (absolute)
        params_rho = solver_params.copy()
        r_bump = 0.01  # 1% absolute rate change
        params_rho['r'] = params_rho['r'] + r_bump
        solver_rho = solver_class(**params_rho)
        result_rho = solver_rho.solve(N_time=100, N_stock=200)
        rho = result_rho.price - base_price  # Per 1% rate change
        
        return {'vega': vega, 'rho': rho}
    
    @staticmethod
    def calculate_all(
        solver_class,
        grid: np.ndarray,
        stock_prices: np.ndarray,
        base_price: float,
        dt: float,
        **solver_params
    ) -> GreeksResult:
        """
        Calculate all Greeks numerically.
        
        Combines grid-based Greeks with bump-and-revalue.
        """
        # From grid
        grid_greeks = NumericalGreeks.calculate_from_grid(
            grid, stock_prices, solver_params['S0'], dt
        )
        
        # Bump-and-revalue
        bump_greeks = NumericalGreeks.calculate_vega_rho(
            solver_class, base_price, **solver_params
        )
        
        return GreeksResult(
            delta=grid_greeks['delta'],
            gamma=grid_greeks['gamma'],
            vega=bump_greeks['vega'],
            theta=grid_greeks['theta'],
            rho=bump_greeks['rho'],
            method='numerical'
        )
