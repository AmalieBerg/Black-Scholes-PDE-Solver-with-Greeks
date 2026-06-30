"""
Comprehensive tests for Black-Scholes PDE solver.

Tests include:
- Convergence to Black-Scholes analytical solution
- Greeks accuracy
- American vs European pricing
- P&L attribution validation
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/claude/Black-Scholes-PDE-Solver/src')

from pde_solver.crank_nicolson import CrankNicolsonSolver
from greeks.greeks_calculator import AnalyticalGreeks, NumericalGreeks
from pnl_attribution.pde_pnl import PDEPnLAttribution, ConvergenceDiagnostics


class TestCrankNicolsonSolver:
    """Test PDE solver accuracy and convergence."""
    
    @pytest.fixture
    def standard_params(self):
        """Standard test parameters."""
        return {
            'S0': 100,
            'K': 100,
            'r': 0.05,
            'q': 0.02,
            'sigma': 0.20,
            'T': 1.0
        }
    
    def test_european_call_convergence(self, standard_params):
        """Test European call converges to Black-Scholes."""
        # Analytical solution
        analytical = AnalyticalGreeks.calculate(
            option_type='call',
            **standard_params
        )
        
        # Black-Scholes formula for price
        from scipy.stats import norm
        S, K, r, q, sigma, T = [standard_params[k] for k in ['S0', 'K', 'r', 'q', 'sigma', 'T']]
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # PDE solution
        solver = CrankNicolsonSolver(option_type='call', exercise_type='european', **standard_params)
        result = solver.solve(N_time=200, N_stock=400)
        
        # Check convergence (within 0.5% for fine grid)
        error = abs(result.price - bs_price)
        error_pct = error / bs_price * 100
        
        assert error_pct < 0.5, f"PDE error {error_pct:.2f}% exceeds 0.5% threshold"
    
    def test_european_put_convergence(self, standard_params):
        """Test European put converges to Black-Scholes."""
        # Analytical solution
        from scipy.stats import norm
        S, K, r, q, sigma, T = [standard_params[k] for k in ['S0', 'K', 'r', 'q', 'sigma', 'T']]
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        
        # PDE solution
        solver = CrankNicolsonSolver(option_type='put', exercise_type='european', **standard_params)
        result = solver.solve(N_time=200, N_stock=400)
        
        error_pct = abs(result.price - bs_price) / bs_price * 100
        assert error_pct < 0.5, f"Put PDE error {error_pct:.2f}% exceeds 0.5%"
    
    def test_american_put_early_exercise(self, standard_params):
        """Test American put is worth more than European (deep ITM)."""
        params = standard_params.copy()
        params['S0'] = 80  # Deep in-the-money put
        
        solver_euro = CrankNicolsonSolver(option_type='put', exercise_type='european', **params)
        solver_amer = CrankNicolsonSolver(option_type='put', exercise_type='american', **params)
        
        euro_result = solver_euro.solve(N_time=100, N_stock=200)
        amer_result = solver_amer.solve(N_time=100, N_stock=200)
        
        # American put should be worth more
        assert amer_result.price > euro_result.price, \
            "American put should be worth more than European for deep ITM"
        
        # Early exercise premium should be reasonable
        premium = amer_result.price - euro_result.price
        assert premium > 0.1, "Early exercise premium too small"
        assert premium < 5.0, "Early exercise premium unreasonably large"


class TestGreeksCalculation:
    """Test Greeks accuracy."""
    
    @pytest.fixture
    def standard_params(self):
        return {
            'S0': 100,
            'K': 100,
            'r': 0.05,
            'q': 0.02,
            'sigma': 0.20,
            'T': 1.0,
            'option_type': 'call'
        }
    
    def test_analytical_greeks_call(self, standard_params):
        """Test analytical Greeks calculation."""
        greeks = AnalyticalGreeks.calculate(**standard_params)
        
        # Delta should be between 0 and 1 for call
        assert 0 < greeks.delta < 1, f"Delta {greeks.delta} out of range"
        
        # Gamma should be positive
        assert greeks.gamma > 0, "Gamma should be positive"
        
        # Vega should be positive
        assert greeks.vega > 0, "Vega should be positive"
        
        # Theta should be negative for long call
        assert greeks.theta < 0, "Theta should be negative for long call"
    
    def test_put_call_parity_greeks(self, standard_params):
        """Test put-call parity for Delta."""
        call_greeks = AnalyticalGreeks.calculate(option_type='call', **standard_params)
        put_greeks = AnalyticalGreeks.calculate(option_type='put', **standard_params)
        
        # Delta_call - Delta_put = exp(-q*T)
        expected_diff = np.exp(-standard_params['q'] * standard_params['T'])
        actual_diff = call_greeks.delta - put_greeks.delta
        
        assert abs(actual_diff - expected_diff) < 0.01, \
            "Put-call parity violated for Delta"
    
    def test_numerical_vs_analytical_greeks(self, standard_params):
        """Compare numerical Greeks from PDE with analytical."""
        solver = CrankNicolsonSolver(exercise_type='european', **standard_params)
        result = solver.solve(N_time=200, N_stock=400)
        
        # Analytical Greeks
        analytical = AnalyticalGreeks.calculate(**standard_params)
        
        # Numerical Greeks from grid
        dt = standard_params['T'] / 200
        numerical_grid = NumericalGreeks.calculate_from_grid(
            result.grid,
            result.stock_prices,
            standard_params['S0'],
            dt
        )
        
        # Delta comparison (within 2%)
        delta_error = abs(numerical_grid['delta'] - analytical.delta) / abs(analytical.delta)
        assert delta_error < 0.02, f"Delta error {delta_error*100:.1f}% too large"
        
        # Gamma comparison (within 5% - second derivative is noisier)
        gamma_error = abs(numerical_grid['gamma'] - analytical.gamma) / abs(analytical.gamma)
        assert gamma_error < 0.05, f"Gamma error {gamma_error*100:.1f}% too large"


class TestPnLAttribution:
    """Test P&L attribution framework."""
    
    @pytest.fixture
    def base_params(self):
        return {
            'S0': 100,
            'K': 100,
            'r': 0.05,
            'q': 0.02,
            'sigma': 0.20,
            'T': 1.0,
            'option_type': 'call',
            'exercise_type': 'european'
        }
    
    def test_clean_pnl_calculation(self, base_params):
        """Test Clean P&L calculation using Greeks."""
        # Get initial Greeks
        greeks_t0 = AnalyticalGreeks.calculate(**base_params)
        
        # Market moves
        market_moves = {
            'dS': 5.0,      # Stock up $5
            'dt': 1/365,    # 1 day
            'dvol': 0.0,
            'dr': 0.0
        }
        
        # Calculate P&L
        pnl_attr = PDEPnLAttribution(CrankNicolsonSolver)
        
        greeks_dict = {
            'delta': greeks_t0.delta,
            'gamma': greeks_t0.gamma,
            'vega': greeks_t0.vega,
            'theta': greeks_t0.theta,
            'rho': greeks_t0.rho
        }
        
        result = pnl_attr.compute_pnl(
            position=100,
            price_t0=10.0,
            price_t1=10.5,
            greeks_t0=greeks_dict,
            market_moves=market_moves
        )
        
        # Clean P&L should be positive (stock up, delta positive)
        assert result.clean_pnl > 0, "Clean P&L should be positive for stock rally"
        
        # Attribution should sum correctly
        total_greeks_pnl = sum([
            result.attribution['delta_pnl'],
            result.attribution['gamma_pnl'],
            result.attribution['vega_pnl'],
            result.attribution['theta_pnl'],
            result.attribution['rho_pnl']
        ])
        
        assert abs(total_greeks_pnl - result.clean_pnl) < 0.01, \
            "Greeks attribution doesn't sum to clean P&L"


class TestConvergence:
    """Test convergence diagnostics."""
    
    def test_convergence_rate(self):
        """Test that Crank-Nicolson achieves O(dt²) convergence."""
        params = {
            'S0': 100,
            'K': 100,
            'r': 0.05,
            'q': 0.02,
            'sigma': 0.20,
            'T': 1.0,
            'option_type': 'call',
            'exercise_type': 'european'
        }
        
        # Analytical price
        from scipy.stats import norm
        S, K, r, q, sigma, T = [params[k] for k in ['S0', 'K', 'r', 'q', 'sigma', 'T']]
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # Test convergence
        diagnostics = ConvergenceDiagnostics()
        conv_results = diagnostics.analyze_convergence(
            CrankNicolsonSolver,
            params,
            bs_price,
            grid_sizes=[(50, 100), (100, 200), (200, 400)]
        )
        
        # Average error ratio should be close to 4 (O(dt²))
        avg_ratio = conv_results['convergence_assessment']['average_error_ratio']
        assert 2.5 < avg_ratio < 6.0, \
            f"Convergence rate {avg_ratio:.2f} not consistent with O(dt²)"


def run_all_tests():
    """Run all tests and report results."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
