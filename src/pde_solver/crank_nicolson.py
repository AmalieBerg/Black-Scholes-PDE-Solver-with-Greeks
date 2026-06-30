"""
Crank-Nicolson finite difference solver for Black-Scholes PDE.

Solves: ∂f/∂t + (r-q-σ²/2)∂f/∂ln(S) + ½σ²∂²f/∂ln(S)² = rf
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PDEResult:
    """Container for PDE solver results."""
    price: float
    grid: np.ndarray
    stock_prices: np.ndarray
    time_steps: np.ndarray
    convergence_error: Optional[float] = None


class CrankNicolsonSolver:
    """
    Crank-Nicolson finite difference solver for Black-Scholes PDE.
    
    Uses log-transformation Z = ln(S) for computational efficiency.
    Solves American and European options with boundary conditions.
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: str = 'call',
        exercise_type: str = 'european'
    ):
        """
        Initialize PDE solver.
        
        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        r : float
            Risk-free rate
        q : float
            Dividend yield
        sigma : float
            Volatility
        T : float
            Time to maturity
        option_type : str
            'call' or 'put'
        exercise_type : str
            'european' or 'american'
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.option_type = option_type.lower()
        self.exercise_type = exercise_type.lower()
        
    def solve(
        self,
        N_time: int = 100,
        N_stock: int = 200,
        S_max: Optional[float] = None
    ) -> PDEResult:
        """
        Solve the Black-Scholes PDE using Crank-Nicolson method.
        
        Parameters:
        -----------
        N_time : int
            Number of time steps
        N_stock : int
            Number of stock price steps (must be even)
        S_max : float, optional
            Maximum stock price for grid
            
        Returns:
        --------
        PDEResult
            Container with price, grid, and diagnostics
        """
        # Setup grid
        dt = self.T / N_time
        
        if S_max is None:
            S_max = self.S0 * np.exp(5 * self.sigma * np.sqrt(self.T))
        
        S_min = self.S0 * np.exp(-5 * self.sigma * np.sqrt(self.T))
        
        # Use log-space for stability
        Z_min = np.log(S_min)
        Z_max = np.log(S_max)
        dZ = (Z_max - Z_min) / N_stock
        
        Z = np.linspace(Z_min, Z_max, N_stock + 1)
        S = np.exp(Z)
        
        # Initialize grid (time x stock)
        V = np.zeros((N_time + 1, N_stock + 1))
        
        # Terminal condition
        if self.option_type == 'call':
            V[-1, :] = np.maximum(S - self.K, 0)
        else:
            V[-1, :] = np.maximum(self.K - S, 0)
        
        # Build Crank-Nicolson matrices
        A, B = self._build_matrices(N_stock, dt, dZ)
        
        # Time-stepping (backward in time)
        for i in range(N_time - 1, -1, -1):
            # Right-hand side
            rhs = B @ V[i + 1, 1:-1]
            
            # Boundary conditions
            rhs[0] += self._lower_boundary(S[0], dt, i, dZ)
            rhs[-1] += self._upper_boundary(S[-1], dt, i, dZ)
            
            # Solve system
            V[i, 1:-1] = spsolve(A, rhs)
            
            # Set boundaries
            V[i, 0] = self._boundary_value(S[0], i * dt)
            V[i, -1] = self._boundary_value(S[-1], i * dt)
            
            # American early exercise check
            if self.exercise_type == 'american':
                if self.option_type == 'call':
                    intrinsic = np.maximum(S - self.K, 0)
                else:
                    intrinsic = np.maximum(self.K - S, 0)
                V[i, :] = np.maximum(V[i, :], intrinsic)
        
        # Interpolate to get price at S0
        price = np.interp(self.S0, S, V[0, :])
        
        return PDEResult(
            price=price,
            grid=V,
            stock_prices=S,
            time_steps=np.linspace(0, self.T, N_time + 1)
        )
    
    def _build_matrices(self, N: int, dt: float, dZ: float) -> tuple:
        """Build Crank-Nicolson coefficient matrices using standard formulation."""
        # PDE coefficients for transformed equation
        mu = self.r - self.q - 0.5 * self.sigma**2
        sig2 = self.sigma**2
        
        # Standard finite difference coefficients
        # V_{i+1,j} = a*V_{i,j-1} + b*V_{i,j} + c*V_{i,j+1}
        
        # For Crank-Nicolson, we have:
        # V^{n} - 0.5*dt*L*V^{n} = V^{n+1} + 0.5*dt*L*V^{n+1}
        # where L is the differential operator
        
        r_dt = self.r * dt
        mu_dt_dZ = mu * dt / (2.0 * dZ)
        sig2_dt_dZ2 = 0.5 * sig2 * dt / (dZ * dZ)
        
        # Coefficients for the system
        a = -0.5 * (sig2_dt_dZ2 - mu_dt_dZ)
        b = 1.0 + sig2_dt_dZ2 + 0.5 * r_dt
        c = -0.5 * (sig2_dt_dZ2 + mu_dt_dZ)
        
        # Build LHS matrix (multiply by this)
        main_A = np.full(N - 1, b)
        upper_A = np.full(N - 2, c)
        lower_A = np.full(N - 2, a)
        
        A = diags([lower_A, main_A, upper_A], [-1, 0, 1], format='csr')
        
        # Build RHS matrix  
        a_rhs = 0.5 * (sig2_dt_dZ2 - mu_dt_dZ)
        b_rhs = 1.0 - sig2_dt_dZ2 - 0.5 * r_dt
        c_rhs = 0.5 * (sig2_dt_dZ2 + mu_dt_dZ)
        
        main_B = np.full(N - 1, b_rhs)
        upper_B = np.full(N - 2, c_rhs)
        lower_B = np.full(N - 2, a_rhs)
        
        B = diags([lower_B, main_B, upper_B], [-1, 0, 1], format='csr')
        
        return A, B
    
    def _boundary_value(self, S: float, t: float) -> float:
        """Calculate boundary value for option."""
        tau = self.T - t
        
        if self.option_type == 'call':
            if S < 0.01 * self.K:  # Lower boundary
                return 0.0
            else:  # Upper boundary
                return S * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau)
        else:  # put
            if S < 0.01 * self.K:  # Lower boundary
                return self.K * np.exp(-self.r * tau)
            else:  # Upper boundary
                return 0.0
    
    def _lower_boundary(self, S: float, dt: float, i: int, dZ: float) -> float:
        """Boundary condition contribution for lower boundary."""
        mu = self.r - self.q - 0.5 * self.sigma**2
        sig2 = self.sigma**2
        
        mu_dt_dZ = mu * dt / (2.0 * dZ)
        sig2_dt_dZ2 = 0.5 * sig2 * dt / (dZ * dZ)
        
        a = -0.5 * (sig2_dt_dZ2 - mu_dt_dZ)
        
        t = i * dt
        V_boundary = self._boundary_value(S, t)
        
        return -a * V_boundary
    
    def _upper_boundary(self, S: float, dt: float, i: int, dZ: float) -> float:
        """Boundary condition contribution for upper boundary."""
        mu = self.r - self.q - 0.5 * self.sigma**2
        sig2 = self.sigma**2
        
        mu_dt_dZ = mu * dt / (2.0 * dZ)
        sig2_dt_dZ2 = 0.5 * sig2 * dt / (dZ * dZ)
        
        c = -0.5 * (sig2_dt_dZ2 + mu_dt_dZ)
        
        t = i * dt
        V_boundary = self._boundary_value(S, t)
        
        return -c * V_boundary
