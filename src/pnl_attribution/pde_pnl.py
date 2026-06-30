"""
P&L Attribution Framework for PDE Solver.

Separates Clean vs Dirty P&L and provides model diagnostics.
This is what separates academic implementations from trading desk reality.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PDEPnLResult:
    """P&L attribution result for PDE solver."""
    clean_pnl: float
    dirty_pnl: float
    residual_pnl: float
    attribution: Dict[str, float]
    warnings: List[str]


class PDEPnLAttribution:
    """
    P&L Attribution for PDE-based pricing.
    
    Decomposes P&L into:
    - Clean P&L: Theoretical price changes (Greeks-based)
    - Dirty P&L: Actual realized P&L including costs
    - Residual P&L: Unexplained P&L indicating model issues
    """
    
    def __init__(self, solver_class):
        """Initialize with PDE solver class."""
        self.solver_class = solver_class
        
    def compute_pnl(
        self,
        position: float,
        price_t0: float,
        price_t1: float,
        greeks_t0: dict,
        market_moves: dict,
        transaction_costs: float = 0.0,
        grid_params_t0: dict = None,
        grid_params_t1: dict = None
    ) -> PDEPnLResult:
        """
        Compute P&L attribution.
        
        Parameters:
        -----------
        position : float
            Number of contracts held
        price_t0 : float
            Option price at t=0
        price_t1 : float
            Option price at t=1
        greeks_t0 : dict
            Greeks at t=0 (delta, gamma, vega, theta, rho)
        market_moves : dict
            Market moves: dS, dt, dvol, dr
        transaction_costs : float
            Trading costs incurred
        grid_params_t0 : dict
            Grid parameters at t=0 (for convergence check)
        grid_params_t1 : dict
            Grid parameters at t=1 (for convergence check)
            
        Returns:
        --------
        PDEPnLResult with attribution breakdown
        """
        # Calculate theoretical P&L (Clean)
        clean_pnl = self._calculate_clean_pnl(
            position, greeks_t0, market_moves
        )
        
        # Calculate actual P&L (Dirty)
        actual_price_change = price_t1 - price_t0
        dirty_pnl = position * actual_price_change - transaction_costs
        
        # Residual P&L
        residual_pnl = dirty_pnl - clean_pnl
        
        # P&L Attribution breakdown
        attribution = {
            'delta_pnl': position * greeks_t0.get('delta', 0) * market_moves.get('dS', 0),
            'gamma_pnl': 0.5 * position * greeks_t0.get('gamma', 0) * market_moves.get('dS', 0)**2,
            'vega_pnl': position * greeks_t0.get('vega', 0) * market_moves.get('dvol', 0),
            'theta_pnl': position * greeks_t0.get('theta', 0) * market_moves.get('dt', 0),
            'rho_pnl': position * greeks_t0.get('rho', 0) * market_moves.get('dr', 0),
            'transaction_costs': -transaction_costs,
            'residual': residual_pnl
        }
        
        # Model diagnostics and warnings
        warnings = self._generate_warnings(
            residual_pnl, dirty_pnl, grid_params_t0, grid_params_t1
        )
        
        return PDEPnLResult(
            clean_pnl=clean_pnl,
            dirty_pnl=dirty_pnl,
            residual_pnl=residual_pnl,
            attribution=attribution,
            warnings=warnings
        )
    
    def _calculate_clean_pnl(
        self,
        position: float,
        greeks: dict,
        moves: dict
    ) -> float:
        """Calculate theoretical P&L using Greeks approximation."""
        delta_pnl = greeks.get('delta', 0) * moves.get('dS', 0)
        gamma_pnl = 0.5 * greeks.get('gamma', 0) * moves.get('dS', 0)**2
        vega_pnl = greeks.get('vega', 0) * moves.get('dvol', 0)
        theta_pnl = greeks.get('theta', 0) * moves.get('dt', 0)
        rho_pnl = greeks.get('rho', 0) * moves.get('dr', 0)
        
        clean_pnl = position * (delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl)
        
        return clean_pnl
    
    def _generate_warnings(
        self,
        residual_pnl: float,
        dirty_pnl: float,
        grid_params_t0: dict,
        grid_params_t1: dict
    ) -> List[str]:
        """
        Generate model risk warnings based on P&L analysis.
        
        Key indicators:
        - Residual P&L > 5% = model failure
        - Grid convergence issues
        - Discretization error patterns
        """
        warnings = []
        
        # Check residual P&L magnitude
        if dirty_pnl != 0:
            residual_pct = abs(residual_pnl / dirty_pnl) * 100
            
            if residual_pct > 5:
                warnings.append(
                    f"  CRITICAL: Residual P&L {residual_pct:.1f}% indicates model failure. "
                    f"Investigate: grid refinement, boundary conditions, or missing risk factors."
                )
            elif residual_pct > 2:
                warnings.append(
                    f"  WARNING: Residual P&L {residual_pct:.1f}% suggests discretization error. "
                    f"Consider refining grid spacing."
                )
        
        # Check grid convergence
        if grid_params_t0 and grid_params_t1:
            if grid_params_t0 != grid_params_t1:
                warnings.append(
                    "  Grid parameters changed between t=0 and t=1. "
                    "This introduces model inconsistency in P&L attribution."
                )
        
        # Check for theta bleed pattern
        if abs(residual_pnl) > 0:
            warnings.append(
                "ℹ  Theta Bleed Diagnostic: Residual P&L may indicate time discretization effects. "
                "For American options, this could signal early exercise boundary approximation errors."
            )
        
        if not warnings:
            warnings.append(" P&L attribution within acceptable ranges. Model performing well.")
        
        return warnings


class ConvergenceDiagnostics:
    """Analyze PDE solver convergence and stability."""
    
    @staticmethod
    def analyze_convergence(
        solver_class,
        solver_params: dict,
        analytical_price: float,
        grid_sizes: List[tuple] = None
    ) -> dict:
        """
        Test convergence by refining grid.
        
        Parameters:
        -----------
        solver_class : class
            PDE solver class
        solver_params : dict
            Solver parameters
        analytical_price : float
            Black-Scholes analytical price for comparison
        grid_sizes : List[tuple]
            List of (N_time, N_stock) pairs to test
            
        Returns:
        --------
        dict with convergence results
        """
        if grid_sizes is None:
            grid_sizes = [(50, 100), (100, 200), (200, 400), (400, 800)]
        
        results = {
            'grid_sizes': [],
            'prices': [],
            'errors': [],
            'error_ratio': []
        }
        
        solver = solver_class(**solver_params)
        
        for N_time, N_stock in grid_sizes:
            result = solver.solve(N_time=N_time, N_stock=N_stock)
            price = result.price
            error = abs(price - analytical_price)
            
            results['grid_sizes'].append((N_time, N_stock))
            results['prices'].append(price)
            results['errors'].append(error)
        
        # Calculate error reduction ratios (should be ~4 for Crank-Nicolson)
        for i in range(1, len(results['errors'])):
            ratio = results['errors'][i-1] / results['errors'][i] if results['errors'][i] > 0 else 0
            results['error_ratio'].append(ratio)
        
        # Convergence quality assessment
        avg_ratio = np.mean(results['error_ratio']) if results['error_ratio'] else 0
        
        results['convergence_assessment'] = {
            'average_error_ratio': avg_ratio,
            'expected_ratio': 4.0,  # O(dt²) for Crank-Nicolson
            'status': 'GOOD' if 3.0 <= avg_ratio <= 5.0 else 'POOR'
        }
        
        return results
    
    @staticmethod
    def stability_check(dt: float, dZ: float, sigma: float) -> dict:
        """
        Check Von Neumann stability condition.
        
        For Crank-Nicolson: unconditionally stable, but accuracy requires:
        dt * sigma² / dZ² < 1 for good accuracy
        """
        nu = 0.5 * sigma**2
        stability_number = dt * nu / dZ**2
        
        return {
            'stability_number': stability_number,
            'is_stable': True,  # Crank-Nicolson always stable
            'accuracy_good': stability_number < 1,
            'recommendation': 'OK' if stability_number < 1 else 'Refine grid for better accuracy'
        }
