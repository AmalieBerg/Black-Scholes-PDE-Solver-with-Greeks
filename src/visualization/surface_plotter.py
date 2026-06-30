"""
Visualization tools for PDE solver results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class PDEVisualizer:
    """Visualization tools for PDE solver results."""
    
    @staticmethod
    def plot_option_surface(
        grid: np.ndarray,
        stock_prices: np.ndarray,
        time_steps: np.ndarray,
        title: str = "Option Price Surface"
    ):
        """Plot 3D surface of option prices over time and stock price."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        T, S = np.meshgrid(time_steps, stock_prices, indexing='ij')
        
        surf = ax.plot_surface(
            S, T, grid,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )
        
        ax.set_xlabel('Stock Price ($)', fontsize=10)
        ax.set_ylabel('Time (years)', fontsize=10)
        ax.set_zlabel('Option Value ($)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_greeks_surface(
        solver_class,
        solver_params: dict,
        greek: str = 'delta',
        S_range: tuple = None,
        T_range: tuple = None,
        n_points: int = 30
    ):
        """
        Plot Greek surface across strike and maturity.
        
        Parameters:
        -----------
        solver_class : class
            PDE solver class
        solver_params : dict
            Base solver parameters
        greek : str
            Which Greek to plot ('delta', 'gamma', 'vega', 'theta')
        S_range : tuple
            (S_min, S_max) range
        T_range : tuple
            (T_min, T_max) range
        n_points : int
            Number of grid points
        """
        from src.greeks.greeks_calculator import AnalyticalGreeks
        
        # Default ranges
        K = solver_params['K']
        if S_range is None:
            S_range = (0.5 * K, 1.5 * K)
        if T_range is None:
            T_range = (0.05, 2.0)
        
        S_values = np.linspace(S_range[0], S_range[1], n_points)
        T_values = np.linspace(T_range[0], T_range[1], n_points)
        
        greek_values = np.zeros((n_points, n_points))
        
        # Calculate Greek at each point
        for i, S in enumerate(S_values):
            for j, T in enumerate(T_values):
                params = solver_params.copy()
                params['S0'] = S
                params['T'] = T
                
                greeks_result = AnalyticalGreeks.calculate(**params)
                greek_values[i, j] = getattr(greeks_result, greek)
        
        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        S_mesh, T_mesh = np.meshgrid(S_values, T_values, indexing='ij')
        
        surf = ax.plot_surface(
            S_mesh, T_mesh, greek_values,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )
        
        ax.set_xlabel('Stock Price ($)', fontsize=10)
        ax.set_ylabel('Time to Maturity (years)', fontsize=10)
        ax.set_zlabel(f'{greek.capitalize()}', fontsize=10)
        ax.set_title(f'{greek.capitalize()} Surface', fontsize=12, fontweight='bold')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_convergence(
        convergence_results: dict,
        save_path: str = None
    ):
        """
        Plot convergence analysis results.
        
        Parameters:
        -----------
        convergence_results : dict
            Results from ConvergenceDiagnostics.analyze_convergence
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        grid_labels = [f"({n},{m})" for n, m in convergence_results['grid_sizes']]
        errors = convergence_results['errors']
        prices = convergence_results['prices']
        
        # Error plot
        ax1.semilogy(range(len(errors)), errors, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Grid Refinement Level', fontsize=11)
        ax1.set_ylabel('Absolute Error (log scale)', fontsize=11)
        ax1.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(grid_labels)))
        ax1.set_xticklabels(grid_labels, rotation=45)
        
        # Add convergence rate
        if convergence_results['error_ratio']:
            avg_ratio = convergence_results['convergence_assessment']['average_error_ratio']
            ax1.text(
                0.02, 0.98, 
                f"Avg Error Ratio: {avg_ratio:.2f}\nExpected (O(dt²)): 4.0",
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9
            )
        
        # Price convergence plot
        ax2.plot(range(len(prices)), prices, 's-', linewidth=2, markersize=8, label='PDE Price')
        ax2.axhline(
            y=prices[-1], color='r', linestyle='--', 
            label=f'Converged: ${prices[-1]:.4f}'
        )
        ax2.set_xlabel('Grid Refinement Level', fontsize=11)
        ax2.set_ylabel('Option Price ($)', fontsize=11)
        ax2.set_title('Price Convergence', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(range(len(grid_labels)))
        ax2.set_xticklabels(grid_labels, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_pnl_attribution(
        pnl_result,
        save_path: str = None
    ):
        """
        Visualize P&L attribution breakdown.
        
        Parameters:
        -----------
        pnl_result : PDEPnLResult
            P&L attribution result
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # P&L waterfall
        components = ['Clean\nP&L', 'Transaction\nCosts', 'Residual\nP&L', 'Dirty\nP&L']
        values = [
            pnl_result.clean_pnl,
            pnl_result.attribution['transaction_costs'],
            pnl_result.residual_pnl,
            pnl_result.dirty_pnl
        ]
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        ax1.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('P&L ($)', fontsize=11)
        ax1.set_title('P&L Attribution Waterfall', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Greeks decomposition
        greeks_pnl = {
            'Delta': pnl_result.attribution['delta_pnl'],
            'Gamma': pnl_result.attribution['gamma_pnl'],
            'Vega': pnl_result.attribution['vega_pnl'],
            'Theta': pnl_result.attribution['theta_pnl'],
            'Rho': pnl_result.attribution['rho_pnl']
        }
        
        greek_names = list(greeks_pnl.keys())
        greek_values = list(greeks_pnl.values())
        greek_colors = ['green' if v >= 0 else 'red' for v in greek_values]
        
        ax2.barh(greek_names, greek_values, color=greek_colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('P&L Contribution ($)', fontsize=11)
        ax2.set_title('Greeks P&L Decomposition', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
