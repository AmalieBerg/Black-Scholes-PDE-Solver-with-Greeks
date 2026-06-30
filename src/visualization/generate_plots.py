"""
Visualization Demo - Generate all plots from surface_plotter.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.pde_solver.crank_nicolson import CrankNicolsonSolver
from src.greeks.greeks_calculator import AnalyticalGreeks
from src.pnl_attribution.pde_pnl import PDEPnLAttribution, ConvergenceDiagnostics
from src.visualization.surface_plotter import PDEVisualizer

print('Generating visualizations...')

# Setup parameters
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

# 1. Option Price Surface
print('1. Creating option price surface...')
solver = CrankNicolsonSolver(**params)
result = solver.solve(N_time=50, N_stock=100)

fig1 = PDEVisualizer.plot_option_surface(
    result.grid,
    result.stock_prices,
    result.time_steps,
    title='European Call Option Price Surface'
)
fig1.savefig('option_price_surface.png', dpi=300, bbox_inches='tight')
print('    Saved: option_price_surface.png')
plt.close(fig1)

# 2. Delta Surface
print('2. Creating Delta surface...')
fig2 = PDEVisualizer.plot_greeks_surface(
    CrankNicolsonSolver,
    params,
    greek='delta',
    S_range=(80, 120),
    T_range=(0.1, 2.0),
    n_points=20
)
fig2.savefig('delta_surface.png', dpi=300, bbox_inches='tight')
print('    Saved: delta_surface.png')
plt.close(fig2)

# 3. Gamma Surface
print('3. Creating Gamma surface...')
fig3 = PDEVisualizer.plot_greeks_surface(
    CrankNicolsonSolver,
    params,
    greek='gamma',
    S_range=(80, 120),
    T_range=(0.1, 2.0),
    n_points=20
)
fig3.savefig('gamma_surface.png', dpi=300, bbox_inches='tight')
print('    Saved: gamma_surface.png')
plt.close(fig3)

# 4. Vega Surface
print('4. Creating Vega surface...')
fig4 = PDEVisualizer.plot_greeks_surface(
    CrankNicolsonSolver,
    params,
    greek='vega',
    S_range=(80, 120),
    T_range=(0.1, 2.0),
    n_points=20
)
fig4.savefig('vega_surface.png', dpi=300, bbox_inches='tight')
print('    Saved: vega_surface.png')
plt.close(fig4)

# 5. Convergence Analysis
print('5. Creating convergence plot...')
from scipy.stats import norm
S, K, r, q, sigma, T = [params[k] for k in ['S0', 'K', 'r', 'q', 'sigma', 'T']]
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

diagnostics = ConvergenceDiagnostics()
conv_results = diagnostics.analyze_convergence(
    CrankNicolsonSolver,
    params,
    bs_price,
    grid_sizes=[(50, 100), (100, 200), (200, 400), (400, 800)]
)

fig5 = PDEVisualizer.plot_convergence(conv_results)
fig5.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
print('    Saved: convergence_analysis.png')
plt.close(fig5)

# 6. P&L Attribution
print('6. Creating P&L attribution plot...')
solver_t0 = CrankNicolsonSolver(**params)
result_t0 = solver_t0.solve(N_time=100, N_stock=200)
greeks_t0 = AnalyticalGreeks.calculate(**params)

market_moves = {
    'dS': 3.0,
    'dt': 1/365,
    'dvol': 0.02,
    'dr': 0.0
}

params_t1 = params.copy()
params_t1['S0'] = params['S0'] + market_moves['dS']
params_t1['T'] = params['T'] - market_moves['dt']
params_t1['sigma'] = params['sigma'] + market_moves['dvol']

solver_t1 = CrankNicolsonSolver(**params_t1)
result_t1 = solver_t1.solve(N_time=100, N_stock=200)

pnl_attr = PDEPnLAttribution(CrankNicolsonSolver)
greeks_dict = {
    'delta': greeks_t0.delta,
    'gamma': greeks_t0.gamma,
    'vega': greeks_t0.vega,
    'theta': greeks_t0.theta,
    'rho': greeks_t0.rho
}

pnl_result = pnl_attr.compute_pnl(
    position=100,
    price_t0=result_t0.price,
    price_t1=result_t1.price,
    greeks_t0=greeks_dict,
    market_moves=market_moves,
    transaction_costs=5.0
)

fig6 = PDEVisualizer.plot_pnl_attribution(pnl_result)
fig6.savefig('pnl_attribution.png', dpi=300, bbox_inches='tight')
print('    Saved: pnl_attribution.png')
plt.close(fig6)

print('\n All visualizations generated successfully!')
print('\nGenerated files:')
print('  - option_price_surface.png')
print('  - delta_surface.png')
print('  - gamma_surface.png')
print('  - vega_surface.png')
print('  - convergence_analysis.png')
print('  - pnl_attribution.png')