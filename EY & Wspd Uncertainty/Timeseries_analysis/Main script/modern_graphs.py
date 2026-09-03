"""

Supporting script for Timeseries analysis.py
Whole job is to make pwetty graphs.

Modern Graph Design Templates
------------------------------
Beautiful, publication-ready visualizations for wind analysis.

All functions return (fig, metadata_dict) for caching.

Usage in your main script:
    from modern_graphs import ModernGraphs

    mg = ModernGraphs()
    fig, meta = mg.scatter_validation(x, y, title="Measured vs Predicted")
    cache.save(fig, location="2023PA032", graph_type="scatter_predicted", metadata=meta)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple, Dict, Optional
import seaborn as sns


class ModernGraphs:
    """
    Modern, publication-ready graph templates.
    """

    def __init__(self):
        """Initialize with modern color scheme and styling."""
        # Professional color palette
        self.colors = {
            "primary": "#2E86AB",  # Blue
            "secondary": "#A23B72",  # Purple
            "success": "#06A77D",  # Green
            "warning": "#F18F01",  # Orange
            "danger": "#C73E1D",  # Red
            "neutral": "#6C757D",  # Gray
            "light": "#E9ECEF",  # Light gray
            "dark": "#212529"  # Dark gray
        }

        # Set default matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.titleweight': 'bold',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
        })

    def scatter_validation(self,
                           x: pd.Series,
                           y: pd.Series,
                           xlabel: str,
                           ylabel: str,
                           title: str,
                           device_info: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, Dict]:
        """
        Modern scatter plot for validation (measured vs predicted).

        Returns:
            (figure, metadata_dict)
        """
        # Calculate statistics
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * x + intercept

        mae = mean_absolute_error(y, x)
        bias = np.mean(x - y)
        stddev = np.std(x - y)
        r2 = r2_score(y, x)
        pearson_r, _ = pearsonr(x, y)
        n = len(x)

        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:, :3])

        # Density-colored scatter
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [self.colors["light"], self.colors["primary"]]
        cmap = LinearSegmentedColormap.from_list("density", colors_list)

        # Calculate point density for coloring
        xy = np.vstack([x, y])
        from scipy.stats import gaussian_kde
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x.iloc[idx], y.iloc[idx], z[idx]

        scatter = ax_main.scatter(x_sorted, y_sorted, c=z_sorted, s=20,
                                  alpha=0.6, cmap=cmap, edgecolors='none')

        # 1:1 line
        max_val = max(max(x), max(y))
        min_val = min(min(x), min(y))
        ax_main.plot([min_val, max_val], [min_val, max_val],
                     'k--', linewidth=1.5, label='1:1 line', alpha=0.7)

        # Fit line
        ax_main.plot(x, line, color=self.colors["danger"],
                     linewidth=2, label='Regression', alpha=0.8)

        # Labels and title
        ax_main.set_xlabel(xlabel, fontweight='bold')
        ax_main.set_ylabel(ylabel, fontweight='bold')
        ax_main.legend(loc='upper left', framealpha=0.9)
        ax_main.grid(True, alpha=0.3)

        # Statistics box (top right of scatter)
        stats_text = (
            f"n = {n:,}\n"
            f"R² = {r2:.4f}\n"
            f"r = {pearson_r:+.4f}\n"
            f"Bias = {bias:+.2f} m/s\n"
            f"MAE = {mae:.2f} m/s\n"
            f"SD = {stddev:.2f} m/s\n"
            f"y = {slope:.3f}x {intercept:+.2f}"
        )

        props = dict(boxstyle='round', facecolor=self.colors["light"],
                     alpha=0.95, edgecolor=self.colors["neutral"], linewidth=1.5)
        ax_main.text(0.97, 0.03, stats_text, transform=ax_main.transAxes,
                     fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                     bbox=props, family='monospace')

        # Marginal histograms
        ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

        # Top histogram (x distribution)
        ax_top.hist(x, bins=50, color=self.colors["primary"],
                    alpha=0.6, edgecolor='white', linewidth=0.5)
        ax_top.set_ylabel('Count', fontsize=9)
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(True, alpha=0.2, axis='y')
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

        # Right histogram (y distribution)
        ax_right.hist(y, bins=50, orientation='horizontal',
                      color=self.colors["secondary"], alpha=0.6,
                      edgecolor='white', linewidth=0.5)
        ax_right.set_xlabel('Count', fontsize=9)
        ax_right.tick_params(labelleft=False)
        ax_right.grid(True, alpha=0.2, axis='x')
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)

        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        if device_info:
            fig.text(0.5, 0.94, device_info, ha='center', fontsize=9,
                     style='italic', color=self.colors["neutral"])

        metadata = {
            "r2": r2,
            "pearson_r": pearson_r,
            "mae": mae,
            "bias": bias,
            "stddev": stddev,
            "slope": slope,
            "intercept": intercept,
            "n_samples": n
        }

        return fig, metadata

    def monthly_correlation_availability(self,
                                         availability: pd.Series,
                                         correlation: pd.Series,
                                         title: str = "Monthly Data Quality",
                                         figsize: Tuple[int, int] = (14, 6)) -> Tuple[plt.Figure, Dict]:
        """
        Combined availability and correlation over time.

        Args:
            availability: Series with datetime index, values in %
            correlation: Series with datetime index, values 0-1

        Returns:
            (figure, metadata_dict)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       facecolor='white', sharex=True)

        # Top: Availability bars
        bars = ax1.bar(availability.index, availability.values,
                       width=20, color=self.colors["primary"],
                       alpha=0.7, edgecolor='white', linewidth=1)

        # Color bars by threshold
        for i, (idx, val) in enumerate(availability.items()):
            if val >= 90:
                bars[i].set_color(self.colors["success"])
            elif val >= 80:
                bars[i].set_color(self.colors["warning"])
            else:
                bars[i].set_color(self.colors["danger"])

        # Add threshold lines
        ax1.axhline(90, color=self.colors["success"], linestyle='--',
                    linewidth=1, alpha=0.5, label='90% threshold')
        ax1.axhline(80, color=self.colors["warning"], linestyle='--',
                    linewidth=1, alpha=0.5, label='80% threshold')

        ax1.set_ylabel('Data Availability (%)', fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')

        # Value labels on bars
        for idx, val in availability.items():
            ax1.text(idx, val + 2, f'{val:.0f}%',
                     ha='center', va='bottom', fontsize=8)

        # Bottom: Correlation line
        ax2.plot(correlation.index, correlation.values,
                 marker='o', markersize=6, linewidth=2,
                 color=self.colors["secondary"], markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=self.colors["secondary"])

        # Fill area under curve
        ax2.fill_between(correlation.index, 0, correlation.values,
                         alpha=0.2, color=self.colors["secondary"])

        # Reference line at r=0.95
        ax2.axhline(0.95, color=self.colors["success"], linestyle='--',
                    linewidth=1, alpha=0.5, label='Target r=0.95')

        ax2.set_ylabel('Correlation Coefficient (r)', fontweight='bold')
        ax2.set_xlabel('Month', fontweight='bold')
        ax2.set_ylim(correlation.min() - 0.02, 1.0)
        ax2.legend(loc='lower right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        metadata = {
            "mean_availability": float(availability.mean()),
            "min_availability": float(availability.min()),
            "mean_correlation": float(correlation.mean()),
            "min_correlation": float(correlation.min()),
            "months_below_80pct": int((availability < 80).sum())
        }

        return fig, metadata

    def mae_by_windspeed(self,
                         df: pd.DataFrame,
                         measured_col: str,
                         predicted_col: str,
                         title: str = "Error Analysis by Wind Speed",
                         figsize: Tuple[int, int] = (12, 5)) -> Tuple[plt.Figure, Dict]:
        """
        Dual-panel: Normalized and absolute MAE by wind speed bins.

        Args:
            df: DataFrame with measured and predicted wind speeds
            measured_col: Name of measured windspeed column
            predicted_col: Name of predicted windspeed column

        Returns:
            (figure, metadata_dict)
        """
        # Filter to >3 m/s for normalized
        df_filtered = df[df[measured_col] > 3].copy()

        # Calculate errors
        df_filtered['abs_error'] = np.abs(df_filtered[predicted_col] - df_filtered[measured_col])
        df_filtered['pct_error'] = (df_filtered['abs_error'] / df_filtered[measured_col]) * 100

        # Create bins
        bin_edges = np.arange(0, df_filtered[measured_col].max() + 0.5, 0.5)
        df_filtered['wind_bin'] = pd.cut(df_filtered[measured_col], bins=bin_edges, right=False)

        # Calculate stats per bin
        stats_pct = df_filtered.groupby('wind_bin', observed=True).agg({
            'pct_error': ['mean', 'std', 'count']
        }).reset_index()
        stats_pct.columns = ['wind_bin', 'mean_pct', 'std_pct', 'count']
        stats_pct['bin_center'] = stats_pct['wind_bin'].apply(lambda x: x.mid)

        stats_abs = df_filtered.groupby('wind_bin', observed=True).agg({
            'abs_error': ['mean', 'std']
        }).reset_index()
        stats_abs.columns = ['wind_bin', 'mean_abs', 'std_abs']
        stats_abs['bin_center'] = stats_abs['wind_bin'].apply(lambda x: x.mid)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

        # Left: Normalized MAE
        ax1.plot(stats_pct['bin_center'], stats_pct['mean_pct'],
                 marker='o', linewidth=2, markersize=6,
                 color=self.colors["primary"], label='Mean error')

        # Error bars (±1 std)
        ax1.fill_between(stats_pct['bin_center'],
                         stats_pct['mean_pct'] - stats_pct['std_pct'],
                         stats_pct['mean_pct'] + stats_pct['std_pct'],
                         alpha=0.2, color=self.colors["primary"], label='±1 SD')

        ax1.set_xlabel('Wind Speed (m/s)', fontweight='bold')
        ax1.set_ylabel('Normalized MAE (%)', fontweight='bold')
        ax1.set_title('Normalized Error', fontweight='bold')
        ax1.legend(framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Right: Absolute MAE
        ax2.plot(stats_abs['bin_center'], stats_abs['mean_abs'],
                 marker='s', linewidth=2, markersize=6,
                 color=self.colors["secondary"], label='Mean error')

        ax2.fill_between(stats_abs['bin_center'],
                         stats_abs['mean_abs'] - stats_abs['std_abs'],
                         stats_abs['mean_abs'] + stats_abs['std_abs'],
                         alpha=0.2, color=self.colors["secondary"], label='±1 SD')

        ax2.set_xlabel('Wind Speed (m/s)', fontweight='bold')
        ax2.set_ylabel('Absolute MAE (m/s)', fontweight='bold')
        ax2.set_title('Absolute Error', fontweight='bold')
        ax2.legend(framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        metadata = {
            "mean_normalized_mae": float(stats_pct['mean_pct'].mean()),
            "mean_absolute_mae": float(stats_abs['mean_abs'].mean()),
            "bins_analyzed": len(stats_pct)
        }

        return fig, metadata

    def executive_dashboard(self,
                            Location_measured: str,
                            stats: Dict,
                            availability: pd.Series,
                            correlation: pd.Series,
                            figsize: Tuple[int, int] = (16, 10)) -> Tuple[plt.Figure, Dict]:
        """
        Single-page executive summary dashboard.

        Args:
            location: Project location name
            stats: Dictionary with key metrics
            availability: Monthly availability series
            correlation: Monthly correlation series

        Returns:
            (figure, metadata_dict)
        """
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Title banner
        fig.suptitle(f'Wind Analysis Summary: {Location_measured}',
                     fontsize=18, fontweight='bold', y=0.98)

        # Top row: Key metrics cards
        metrics = [
            ("Correlation", stats.get('pearson_r', 0), 'r', 0.95),
            ("R² Score", stats.get('r2', 0), 'R²', 0.90),
            ("MAE", stats.get('mae', 0), 'm/s', None),
        ]

        for i, (label, value, unit, target) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')

            # Determine color
            if target is not None:
                if value >= target:
                    color = self.colors["success"]
                elif value >= target * 0.9:
                    color = self.colors["warning"]
                else:
                    color = self.colors["danger"]
            else:
                color = self.colors["neutral"]

            # Draw card
            rect = Rectangle((0.1, 0.2), 0.8, 0.6,
                             facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=3)
            ax.add_patch(rect)

            # Value
            ax.text(0.5, 0.6, f'{value:.3f}',
                    ha='center', va='center', fontsize=32,
                    fontweight='bold', color=color)

            # Label
            ax.text(0.5, 0.35, f'{label}\n({unit})',
                    ha='center', va='center', fontsize=11,
                    color=self.colors["dark"])

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Middle row: Time series
        ax_avail = fig.add_subplot(gs[1, :2])
        ax_avail.bar(availability.index, availability.values,
                     width=20, color=self.colors["primary"], alpha=0.6)
        ax_avail.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax_avail.set_ylabel('Availability (%)', fontweight='bold')
        ax_avail.set_title('Monthly Data Availability', fontweight='bold')
        ax_avail.grid(True, alpha=0.3, axis='y')

        ax_corr = fig.add_subplot(gs[1, 2])
        ax_corr.plot(correlation.index, correlation.values,
                     marker='o', linewidth=2, color=self.colors["secondary"])
        ax_corr.fill_between(correlation.index, 0, correlation.values,
                             alpha=0.3, color=self.colors["secondary"])
        ax_corr.set_ylabel('Correlation', fontweight='bold')
        ax_corr.set_title('Monthly Correlation', fontweight='bold')
        ax_corr.grid(True, alpha=0.3)

        # Bottom row: Summary table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')

        table_data = [
            ['Device (Measured)', stats.get('device_measured', 'N/A')],
            ['Device (Predicted)', stats.get('device_predicted', 'N/A')],
            ['Distance', f"{stats.get('distance_m', 0):.0f} m"],
            ['Height Difference', f"{stats.get('dz', 0):.1f} m"],
            ['T-RIX', f"{stats.get('trix', 0):.1f}%"],
            ['Energy Deviation', f"{stats.get('energy_deviation', 0) * 100:+.2f}%"],
            ['Sample Count', f"{stats.get('n_samples', 0):,}"],
            ['Bias', f"{stats.get('bias', 0):+.2f} m/s"],
        ]

        table = ax_table.table(cellText=table_data,
                               colLabels=['Metric', 'Value'],
                               cellLoc='left',
                               loc='center',
                               bbox=[0.2, 0.1, 0.6, 0.8])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor(self.colors["primary"])
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor(self.colors["light"])

        plt.tight_layout()

        metadata = stats.copy()
        return fig, metadata


# Convenience function
def create_modern_graphs() -> ModernGraphs:
    """Factory function to create ModernGraphs instance."""
    return ModernGraphs()