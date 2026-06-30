"""
Comprehensive example demonstrating Black-Scholes PDE Solver.

Demonstrates:
1. European and American option pricing
2. Greeks calculation (analytical and numerical)
3. Volatility surface visualization
4. P&L attribution analysis
5. Convergence diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pde_solver.crank_nicolson import CrankNicolsonSolver
from src.greeks.greeks_calculator import AnalyticalGreeks, NumericalGreeks
from src.pnl_attribution.pde_pnl import PDEPnLAttribution, ConvergenceDiagnostics
from src.visualization.surface_plotter import PDEVisualizer


def example_1_basic_pricing():
    """Example 1: Basic option pricing."""
    print("=" * 60)
    print("EXAMPLE 1: Basic European Call Option Pricing")
    print("=" * 60)
    
    # Setup
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
    
    # Solve PDE
    solver = CrankNicolsonSolver(**params)
    result = solver.solve(N_time=100, N_stock=200)
    
    # Analytical comparison
    from scipy.stats import norm
    S, K, r, q, sigma, T = [params[k] for k in ['S0', 'K', 'r', 'q', 'sigma', 'T']]
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    print(f"\nPDE Price:        ${result.price:.4f}")
    print(f"Black-Scholes:    ${bs_price:.4f}")
    print(f"Error:            ${abs(result.price - bs_price):.4f} ({abs(result.price-bs_price)/bs_price*100:.2f}%)")
    print("\n" + "=" * 60 + "\n")


def example_2_greeks_comparison():
    """Example 2: Greeks calculation and comparison."""
    print("=" * 60)
    print("EXAMPLE 2: Greeks Calculation (Analytical vs Numerical)")
    print("=" * 60)
    
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
    
    # Analytical Greeks
    analytical = AnalyticalGreeks.calculate(**params)
    
    # Numerical Greeks from PDE
    solver = CrankNicolsonSolver(**params)
    result = solver.solve(N_time=200, N_stock=400)
    
    numerical = NumericalGreeks.calculate_all(
        CrankNicolsonSolver,
        result.grid,
        result.stock_prices,
        result.price,
        params['T'] / 200,
        **params
    )
    
    print("\n{:<15} {:<15} {:<15} {:<10}".format("Greek", "Analytical", "Numerical", "Error %"))
    print("-" * 60)
    
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        anal_val = getattr(analytical, greek)
        num_val = getattr(numerical, greek)
        error = abs(anal_val - num_val) / abs(anal_val) * 100 if anal_val != 0 else 0
        
        print(f"{greek.capitalize():<15} {anal_val:>14.6f} {num_val:>14.6f} {error:>9.2f}%")
    
    print("\n" + "=" * 60 + "\n")


def example_3_american_vs_european():
    """Example 3: American vs European put option."""
    print("=" * 60)
    print("EXAMPLE 3: American vs European Put Option (Deep ITM)")
    print("=" * 60)
    
    params = {
        'S0': 80,  # Deep in-the-money
        'K': 100,
        'r': 0.05,
        'q': 0.02,
        'sigma': 0.25,
        'T': 1.0,
        'option_type': 'put'
    }
    
    # European put
    solver_euro = CrankNicolsonSolver(exercise_type='european', **params)
    result_euro = solver_euro.solve(N_time=100, N_stock=200)
    
    # American put
    solver_amer = CrankNicolsonSolver(exercise_type='american', **params)
    result_amer = solver_amer.solve(N_time=100, N_stock=200)
    
    print(f"\nEuropean Put:        ${result_euro.price:.4f}")
    print(f"American Put:        ${result_amer.price:.4f}")
    print(f"Early Exercise Premium: ${result_amer.price - result_euro.price:.4f}")
    print(f"Intrinsic Value:     ${max(params['K'] - params['S0'], 0):.4f}")
    
    print("\n" + "=" * 60 + "\n")


def example_4_pnl_attribution():
    """Example 4: P&L attribution analysis."""
    print("=" * 60)
    print("EXAMPLE 4: P&L Attribution Analysis")
    print("=" * 60)
    
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
    
    # Initial state (t=0)
    solver_t0 = CrankNicolsonSolver(**params)
    result_t0 = solver_t0.solve(N_time=100, N_stock=200)
    greeks_t0 = AnalyticalGreeks.calculate(**params)
    
    # Market moves after 1 day
    market_moves = {
        'dS': 3.0,      # Stock up $3
        'dt': 1/365,    # 1 day
        'dvol': 0.02,   # Vol up 2%
        'dr': 0.0       # No rate change
    }
    
    # New state (t=1)
    params_t1 = params.copy()
    params_t1['S0'] = params['S0'] + market_moves['dS']
    params_t1['T'] = params['T'] - market_moves['dt']
    params_t1['sigma'] = params['sigma'] + market_moves['dvol']
    
    solver_t1 = CrankNicolsonSolver(**params_t1)
    result_t1 = solver_t1.solve(N_time=100, N_stock=200)
    
    # P&L attribution
    pnl_attr = PDEPnLAttribution(CrankNicolsonSolver)
    
    greeks_dict = {
        'delta': greeks_t0.delta,
        'gamma': greeks_t0.gamma,
        'vega': greeks_t0.vega,
        'theta': greeks_t0.theta,
        'rho': greeks_t0.rho
    }
    
    pnl_result = pnl_attr.compute_pnl(
        position=100,  # Long 100 calls
        price_t0=result_t0.price,
        price_t1=result_t1.price,
        greeks_t0=greeks_dict,
        market_moves=market_moves,
        transaction_costs=5.0  # $5 in costs
    )
    
    print(f"\nP&L Attribution for 100 Long Calls:")
    print(f"Initial Price:    ${result_t0.price:.4f}")
    print(f"Final Price:      ${result_t1.price:.4f}")
    print(f"Price Change:     ${result_t1.price - result_t0.price:.4f}")
    print("\nP&L Breakdown:")
    print(f"  Clean P&L:      ${pnl_result.clean_pnl:.2f}")
    print(f"  Dirty P&L:      ${pnl_result.dirty_pnl:.2f}")
    print(f"  Residual P&L:   ${pnl_result.residual_pnl:.2f}")
    
    print("\nGreeks Attribution:")
    for greek, value in pnl_result.attribution.items():
        if 'pnl' in greek:
            print(f"  {greek.replace('_pnl', '').capitalize():<10}: ${value:>8.2f}")
    
    print("\nModel Diagnostics:")
    for warning in pnl_result.warnings:
        print(f"  {warning}")
    
    print("\n" + "=" * 60 + "\n")


def example_5_convergence_analysis():
    """Example 5: Convergence analysis."""
    print("=" * 60)
    print("EXAMPLE 5: Convergence Analysis")
    print("=" * 60)
    
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
    
    # Convergence test
    diagnostics = ConvergenceDiagnostics()
    conv_results = diagnostics.analyze_convergence(
        CrankNicolsonSolver,
        params,
        bs_price,
        grid_sizes=[(50, 100), (100, 200), (200, 400), (400, 800)]
    )
    
    print(f"\nAnalytical Price: ${bs_price:.6f}")
    print(f"\n{'Grid Size':<15} {'PDE Price':<15} {'Error':<15} {'Error Ratio'}")
    print("-" * 70)
    
    for i, (grid_size, price, error) in enumerate(zip(
        conv_results['grid_sizes'],
        conv_results['prices'],
        conv_results['errors']
    )):
        ratio = conv_results['error_ratio'][i] if i < len(conv_results['error_ratio']) else 0
        print(f"{str(grid_size):<15} ${price:<14.6f} ${error:<14.6f} {ratio:>10.2f}")
    
    assessment = conv_results['convergence_assessment']
    print(f"\nConvergence Assessment:")
    print(f"  Average Error Ratio: {assessment['average_error_ratio']:.2f}")
    print(f"  Expected Ratio:      {assessment['expected_ratio']:.2f} (O(dt²))")
    print(f"  Status:              {assessment['status']}")
    
    print("\n" + "=" * 60 + "\n")


def run_all_examples():
    """Run all examples."""
    example_1_basic_pricing()
    example_2_greeks_comparison()
    example_3_american_vs_european()
    example_4_pnl_attribution()
    example_5_convergence_analysis()
    
    print("All examples completed successfully!")


if __name__ == '__main__':
    run_all_examples()
