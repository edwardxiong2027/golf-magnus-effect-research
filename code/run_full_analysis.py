"""
Comprehensive Analysis Script for Magnus Effect Research

This script runs all simulations and generates data/figures for the research paper.
Calibrated model validated against PGA Tour TrackMan data.

Author: High School Research Project
Date: February 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from magnus_simulation import (
    GolfBallSimulator, LaunchConditions,
    mph_to_ms, meters_to_yards, ms_to_mph
)

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Create directories
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def save_figure(fig, name):
    """Save figure to figures directory."""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png")
    plt.close(fig)


# =============================================================================
# SECTION 1: PGA TOUR REFERENCE DATA
# =============================================================================

def create_pga_tour_data():
    """Create PGA Tour reference dataset from TrackMan averages."""
    print("\n" + "="*60)
    print("SECTION 1: Creating PGA Tour Reference Data")
    print("="*60)

    # PGA Tour TrackMan averages (2023-2024 season data)
    data = {
        'club': ['Driver', '3-Wood', '5-Wood', '3-Hybrid', '3-Iron', '4-Iron',
                 '5-Iron', '6-Iron', '7-Iron', '8-Iron', '9-Iron', 'PW'],
        'ball_speed_mph': [171, 158, 152, 146, 142, 137, 132, 127, 120, 115, 109, 102],
        'launch_angle_deg': [10.4, 9.3, 9.7, 10.2, 10.4, 11.0, 12.1, 14.1, 16.3, 18.1, 20.4, 24.2],
        'spin_rate_rpm': [2545, 3655, 4350, 4437, 4630, 4801, 5361, 6231, 7097, 7998, 8647, 9304],
        'carry_yards': [275, 243, 230, 222, 212, 203, 194, 183, 172, 160, 148, 136],
        'max_height_yards': [32, 30, 31, 31, 30, 31, 31, 32, 33, 33, 33, 31],
        'land_angle_deg': [38, 42, 44, 45, 46, 47, 48, 49, 50, 50, 51, 52]
    }

    df = pd.DataFrame(data)
    csv_path = os.path.join(DATA_DIR, 'pga_tour_trackman_averages.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Created PGA Tour dataset with {len(df)} clubs")

    return df


# =============================================================================
# SECTION 2: MODEL VALIDATION
# =============================================================================

def run_model_validation(pga_data):
    """Validate model against PGA Tour data."""
    print("\n" + "="*60)
    print("SECTION 2: Model Validation Against PGA Tour Data")
    print("="*60)

    sim = GolfBallSimulator()
    predictions = []

    for _, row in pga_data.iterrows():
        conditions = LaunchConditions(
            ball_speed=mph_to_ms(row['ball_speed_mph']),
            launch_angle=row['launch_angle_deg'],
            launch_direction=0,
            spin_rate=row['spin_rate_rpm'],
            spin_axis=0,
            altitude=0,
            temperature=20
        )

        sim.simulate(conditions)

        predictions.append({
            'club': row['club'],
            'actual_carry': row['carry_yards'],
            'predicted_carry': meters_to_yards(sim.get_carry_distance()),
            'actual_height': row['max_height_yards'],
            'predicted_height': meters_to_yards(sim.get_max_height()),
            'actual_land_angle': row['land_angle_deg'],
            'predicted_land_angle': sim.get_land_angle(),
            'flight_time': sim.get_flight_time()
        })

    results = pd.DataFrame(predictions)
    results['carry_error'] = results['predicted_carry'] - results['actual_carry']
    results['carry_error_pct'] = (results['carry_error'] / results['actual_carry']) * 100
    results['height_error'] = results['predicted_height'] - results['actual_height']

    # Save results
    csv_path = os.path.join(DATA_DIR, 'model_validation_results.csv')
    results.to_csv(csv_path, index=False)

    # Calculate metrics
    mae = results['carry_error'].abs().mean()
    mape = results['carry_error_pct'].abs().mean()
    rmse = np.sqrt((results['carry_error']**2).mean())
    r_value = stats.pearsonr(results['actual_carry'], results['predicted_carry'])[0]
    r_squared = r_value**2

    print(f"\n  Validation Metrics:")
    print(f"    R² = {r_squared:.4f}")
    print(f"    Mean Absolute Error = {mae:.1f} yards")
    print(f"    RMSE = {rmse:.1f} yards")
    print(f"    Mean Absolute % Error = {mape:.1f}%")

    # Create validation figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter plot: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(results['actual_carry'], results['predicted_carry'],
                s=100, c='steelblue', alpha=0.8, edgecolors='white', linewidth=1.5)

    # Perfect prediction line
    min_val, max_val = 120, 290
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    # ±5% error bands
    ax1.fill_between([min_val, max_val],
                     [min_val*0.95, max_val*0.95],
                     [min_val*1.05, max_val*1.05],
                     alpha=0.2, color='green', label='±5% Error Band')

    # Labels
    for _, row in results.iterrows():
        offset = (5, 5) if row['carry_error'] >= 0 else (5, -12)
        ax1.annotate(row['club'], (row['actual_carry'], row['predicted_carry']),
                    textcoords='offset points', xytext=offset, fontsize=9)

    ax1.set_xlabel('Actual Carry Distance (yards)', fontsize=12)
    ax1.set_ylabel('Predicted Carry Distance (yards)', fontsize=12)
    ax1.set_title(f'Model Validation: R² = {r_squared:.3f}, MAE = {mae:.1f} yards', fontsize=13)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Error bar chart
    ax2 = axes[1]
    colors = ['#2ecc71' if abs(e) < 5 else '#f39c12' if abs(e) < 10 else '#e74c3c'
              for e in results['carry_error']]
    bars = ax2.bar(range(len(results)), results['carry_error'], color=colors,
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=1.5)
    ax2.axhline(y=5, color='#f39c12', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=-5, color='#f39c12', linestyle='--', alpha=0.7, linewidth=1)

    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(results['club'], rotation=45, ha='right', fontsize=10)
    ax2.set_xlabel('Club', fontsize=12)
    ax2.set_ylabel('Prediction Error (yards)', fontsize=12)
    ax2.set_title(f'Error by Club (MAPE = {mape:.1f}%)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'fig1_model_validation')

    return results, {'R2': r_squared, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# =============================================================================
# SECTION 3: SPIN RATE ANALYSIS
# =============================================================================

def run_spin_rate_analysis():
    """Analyze effect of spin rate on carry distance."""
    print("\n" + "="*60)
    print("SECTION 3: Spin Rate vs Carry Distance Analysis")
    print("="*60)

    sim = GolfBallSimulator()

    # Different ball speeds to analyze
    ball_speeds = [150, 160, 170, 180]  # mph
    spin_rates = np.arange(1000, 5001, 100)  # rpm

    all_results = []

    for speed in ball_speeds:
        print(f"  Simulating ball speed: {speed} mph...")
        for spin in spin_rates:
            conditions = LaunchConditions(
                ball_speed=mph_to_ms(speed),
                launch_angle=11.0,
                launch_direction=0,
                spin_rate=spin,
                spin_axis=0
            )

            sim.simulate(conditions)

            all_results.append({
                'ball_speed_mph': speed,
                'spin_rate_rpm': spin,
                'carry_yards': meters_to_yards(sim.get_carry_distance()),
                'max_height_yards': meters_to_yards(sim.get_max_height()),
                'flight_time_s': sim.get_flight_time(),
                'land_angle_deg': sim.get_land_angle()
            })

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(DATA_DIR, 'spin_rate_analysis.csv')
    df.to_csv(csv_path, index=False)

    # Find optimal spin for each ball speed
    optimal_spins = {}
    for speed in ball_speeds:
        subset = df[df['ball_speed_mph'] == speed]
        max_idx = subset['carry_yards'].idxmax()
        optimal_spins[speed] = int(df.loc[max_idx, 'spin_rate_rpm'])
        max_carry = df.loc[max_idx, 'carry_yards']
        print(f"    {speed} mph: Optimal spin = {optimal_spins[speed]} rpm, Max carry = {max_carry:.1f} yards")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Carry vs Spin Rate curves
    ax1 = axes[0]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, speed in enumerate(ball_speeds):
        subset = df[df['ball_speed_mph'] == speed]
        ax1.plot(subset['spin_rate_rpm'], subset['carry_yards'],
                linewidth=2.5, color=colors[i], label=f'{speed} mph')

        # Mark optimal point
        opt_spin = optimal_spins[speed]
        opt_row = subset[subset['spin_rate_rpm'] == opt_spin].iloc[0]
        ax1.scatter([opt_spin], [opt_row['carry_yards']], color=colors[i], s=120,
                   marker='*', edgecolors='black', linewidths=1, zorder=5)

    # Add PGA Tour average zone
    ax1.axvspan(2400, 2700, alpha=0.15, color='gold', label='PGA Tour Driver Range')

    ax1.set_xlabel('Backspin Rate (rpm)', fontsize=12)
    ax1.set_ylabel('Carry Distance (yards)', fontsize=12)
    ax1.set_title('Effect of Backspin on Carry Distance', fontsize=13)
    ax1.legend(title='Ball Speed', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1000, 5000)

    # Right: Height vs Spin Rate
    ax2 = axes[1]
    for i, speed in enumerate(ball_speeds):
        subset = df[df['ball_speed_mph'] == speed]
        ax2.plot(subset['spin_rate_rpm'], subset['max_height_yards'],
                linewidth=2.5, color=colors[i], label=f'{speed} mph')

    ax2.set_xlabel('Backspin Rate (rpm)', fontsize=12)
    ax2.set_ylabel('Maximum Height (yards)', fontsize=12)
    ax2.set_title('Effect of Backspin on Ball Flight Apex', fontsize=13)
    ax2.legend(title='Ball Speed', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1000, 5000)

    plt.suptitle('Magnus Effect: Backspin Influence on Driver Performance',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig2_spin_rate_analysis')

    return df, optimal_spins


# =============================================================================
# SECTION 4: SPIN AXIS (DRAW/FADE) ANALYSIS
# =============================================================================

def run_spin_axis_analysis():
    """Analyze effect of spin axis on lateral deviation."""
    print("\n" + "="*60)
    print("SECTION 4: Spin Axis Effect Analysis (Draw/Fade)")
    print("="*60)

    sim = GolfBallSimulator()

    # Base driver conditions
    base_speed = mph_to_ms(170)
    base_launch = 11.0
    base_spin = 2500

    # Vary spin axis from -25 (draw) to +25 (fade)
    spin_axes = np.arange(-25, 26, 1)

    results = []
    for axis in spin_axes:
        conditions = LaunchConditions(
            ball_speed=base_speed,
            launch_angle=base_launch,
            launch_direction=0,
            spin_rate=base_spin,
            spin_axis=axis
        )

        sim.simulate(conditions)

        results.append({
            'spin_axis_deg': axis,
            'lateral_deviation_yards': meters_to_yards(sim.get_lateral_deviation()),
            'carry_yards': meters_to_yards(sim.get_carry_distance())
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(DATA_DIR, 'spin_axis_analysis.csv')
    df.to_csv(csv_path, index=False)

    # Linear regression for sensitivity
    slope, intercept, r, p, se = stats.linregress(
        df['spin_axis_deg'], df['lateral_deviation_yards'])
    sensitivity = abs(slope)

    print(f"  Spin Axis Sensitivity: {sensitivity:.2f} yards per degree")
    print(f"  Linear fit R² = {r**2:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Lateral deviation vs spin axis
    ax1 = axes[0]
    ax1.plot(df['spin_axis_deg'], df['lateral_deviation_yards'],
            'b-', linewidth=2.5)

    # Shade regions
    ax1.fill_between(df['spin_axis_deg'], 0, df['lateral_deviation_yards'],
                    where=df['lateral_deviation_yards'] > 0,
                    alpha=0.3, color='#3498db', label='Fade (Right)')
    ax1.fill_between(df['spin_axis_deg'], 0, df['lateral_deviation_yards'],
                    where=df['lateral_deviation_yards'] < 0,
                    alpha=0.3, color='#e74c3c', label='Draw (Left)')

    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.axvline(x=0, color='black', linewidth=1)

    # Add linear fit line
    fit_x = np.array([-25, 25])
    fit_y = slope * fit_x + intercept
    ax1.plot(fit_x, fit_y, 'k--', linewidth=1.5, alpha=0.7,
            label=f'Linear fit: {slope:.2f} yds/deg')

    ax1.set_xlabel('Spin Axis (degrees)', fontsize=12)
    ax1.set_ylabel('Lateral Deviation (yards)', fontsize=12)
    ax1.set_title('Spin Axis Effect on Ball Curvature', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Carry distance vs spin axis
    ax2 = axes[1]
    ax2.plot(df['spin_axis_deg'], df['carry_yards'], 'g-', linewidth=2.5)
    ax2.axvline(x=0, color='black', linewidth=1)

    # Calculate distance loss
    max_carry = df['carry_yards'].max()
    carry_at_20 = df[df['spin_axis_deg'] == 20]['carry_yards'].values[0]
    distance_loss = max_carry - carry_at_20

    ax2.annotate(f'{distance_loss:.1f} yard loss\nat ±20° axis',
                xy=(20, carry_at_20), xytext=(5, carry_at_20 - 8),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='black'))

    ax2.set_xlabel('Spin Axis (degrees)', fontsize=12)
    ax2.set_ylabel('Carry Distance (yards)', fontsize=12)
    ax2.set_title('Carry Distance Loss from Sidespin', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Magnus Effect: Spin Axis Influence on Shot Shape (170 mph, 2500 rpm)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig3_spin_axis_analysis')

    return df, sensitivity


# =============================================================================
# SECTION 5: TRAJECTORY VISUALIZATION
# =============================================================================

def create_trajectory_visualizations():
    """Create trajectory plots for different shot shapes."""
    print("\n" + "="*60)
    print("SECTION 5: Creating Trajectory Visualizations")
    print("="*60)

    sim = GolfBallSimulator()

    # Shot types to visualize
    shot_types = [
        {'name': 'Straight', 'axis': 0, 'color': '#2ecc71'},
        {'name': 'Draw (-8°)', 'axis': -8, 'color': '#e74c3c'},
        {'name': 'Fade (+8°)', 'axis': 8, 'color': '#3498db'},
        {'name': 'Hook (-15°)', 'axis': -15, 'color': '#c0392b'},
        {'name': 'Slice (+15°)', 'axis': 15, 'color': '#2980b9'}
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    trajectories = {}

    for shot in shot_types:
        conditions = LaunchConditions(
            ball_speed=mph_to_ms(170),
            launch_angle=11.0,
            launch_direction=0,
            spin_rate=2500,
            spin_axis=shot['axis']
        )

        sim.simulate(conditions)
        trajectories[shot['name']] = {
            'z': [meters_to_yards(p.z) for p in sim.trajectory],
            'y': [meters_to_yards(p.y) for p in sim.trajectory],
            'x': [meters_to_yards(p.x) for p in sim.trajectory]
        }

        # Side view
        axes[0].plot(trajectories[shot['name']]['z'],
                    trajectories[shot['name']]['y'],
                    color=shot['color'], linewidth=2, label=shot['name'])

        # Top view
        axes[1].plot(trajectories[shot['name']]['z'],
                    trajectories[shot['name']]['x'],
                    color=shot['color'], linewidth=2, label=shot['name'])

    # Format side view
    axes[0].set_xlabel('Distance (yards)', fontsize=12)
    axes[0].set_ylabel('Height (yards)', fontsize=12)
    axes[0].set_title('Side View - Ball Flight Height', fontsize=13)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Format top view
    axes[1].set_xlabel('Distance (yards)', fontsize=12)
    axes[1].set_ylabel('Lateral Deviation (yards)', fontsize=12)
    axes[1].set_title('Top View - Ball Curvature', fontsize=13)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Driver Shot Trajectory Comparison (170 mph, 2500 rpm)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig4_trajectory_comparison')

    # 3D trajectory
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    for shot in shot_types[:3]:  # Just straight, draw, fade for clarity
        traj = trajectories[shot['name']]
        ax.plot(traj['z'], traj['x'], traj['y'],
               color=shot['color'], linewidth=2, label=shot['name'])

    ax.set_xlabel('Distance (yards)')
    ax.set_ylabel('Lateral (yards)')
    ax.set_zlabel('Height (yards)')
    ax.set_title('3D Trajectory: Straight, Draw, and Fade', fontsize=13)
    ax.legend(fontsize=10)

    save_figure(fig, 'fig5_3d_trajectory')

    return trajectories


# =============================================================================
# SECTION 6: ENVIRONMENTAL EFFECTS
# =============================================================================

def run_environmental_analysis():
    """Analyze effects of altitude and temperature."""
    print("\n" + "="*60)
    print("SECTION 6: Environmental Effects Analysis")
    print("="*60)

    sim = GolfBallSimulator()

    # Base conditions
    base_conditions = LaunchConditions(
        ball_speed=mph_to_ms(170),
        launch_angle=11.0,
        launch_direction=0,
        spin_rate=2500,
        spin_axis=0,
        altitude=0,
        temperature=20
    )

    # Altitude analysis (0 to 7000 feet)
    altitudes_ft = np.arange(0, 7001, 250)
    altitude_results = []

    for alt_ft in altitudes_ft:
        alt_m = alt_ft * 0.3048
        conditions = LaunchConditions(
            ball_speed=base_conditions.ball_speed,
            launch_angle=base_conditions.launch_angle,
            launch_direction=0,
            spin_rate=base_conditions.spin_rate,
            spin_axis=0,
            altitude=alt_m,
            temperature=20
        )

        sim.simulate(conditions)
        altitude_results.append({
            'altitude_ft': alt_ft,
            'carry_yards': meters_to_yards(sim.get_carry_distance()),
            'air_density': sim.get_air_density(alt_m, 20)
        })

    alt_df = pd.DataFrame(altitude_results)

    # Temperature analysis (30°F to 100°F)
    temperatures_f = np.arange(30, 101, 5)
    temp_results = []

    for temp_f in temperatures_f:
        temp_c = (temp_f - 32) * 5/9
        conditions = LaunchConditions(
            ball_speed=base_conditions.ball_speed,
            launch_angle=base_conditions.launch_angle,
            launch_direction=0,
            spin_rate=base_conditions.spin_rate,
            spin_axis=0,
            altitude=0,
            temperature=temp_c
        )

        sim.simulate(conditions)
        temp_results.append({
            'temperature_F': temp_f,
            'temperature_C': temp_c,
            'carry_yards': meters_to_yards(sim.get_carry_distance())
        })

    temp_df = pd.DataFrame(temp_results)

    # Save data
    alt_df.to_csv(os.path.join(DATA_DIR, 'altitude_effects.csv'), index=False)
    temp_df.to_csv(os.path.join(DATA_DIR, 'temperature_effects.csv'), index=False)

    # Calculate effects
    baseline = alt_df[alt_df['altitude_ft'] == 0]['carry_yards'].values[0]
    denver = alt_df[alt_df['altitude_ft'] == 5000]['carry_yards'].values[0]
    alt_effect = denver - baseline

    cold = temp_df[temp_df['temperature_F'] == 40]['carry_yards'].values[0]
    hot = temp_df[temp_df['temperature_F'] == 90]['carry_yards'].values[0]
    temp_effect = hot - cold

    print(f"  Altitude effect: +{alt_effect:.1f} yards at 5000 ft (Denver)")
    print(f"  Temperature effect: +{temp_effect:.1f} yards (90°F vs 40°F)")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Altitude effect
    ax1 = axes[0]
    ax1.plot(alt_df['altitude_ft'], alt_df['carry_yards'], 'b-', linewidth=2.5)
    ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Sea level: {baseline:.0f} yards')

    # Mark Denver
    ax1.axvline(x=5280, color='red', linestyle='--', alpha=0.7)
    ax1.scatter([5280], [denver + 1], color='red', s=50, zorder=5)
    ax1.annotate(f'Denver (5,280 ft)\n+{alt_effect:.1f} yards',
                xy=(5280, denver), xytext=(5500, denver - 3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    ax1.set_xlabel('Altitude (feet)', fontsize=12)
    ax1.set_ylabel('Carry Distance (yards)', fontsize=12)
    ax1.set_title('Effect of Altitude on Carry Distance', fontsize=13)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Temperature effect
    ax2 = axes[1]
    ax2.plot(temp_df['temperature_F'], temp_df['carry_yards'], 'r-', linewidth=2.5)
    ax2.axvline(x=70, color='green', linestyle='--', alpha=0.7, label='Standard temp (70°F)')

    ax2.set_xlabel('Temperature (°F)', fontsize=12)
    ax2.set_ylabel('Carry Distance (yards)', fontsize=12)
    ax2.set_title('Effect of Temperature on Carry Distance', fontsize=13)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Environmental Effects on Driver Distance (170 mph, 2500 rpm)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig6_environmental_effects')

    return alt_df, temp_df, {'altitude_effect': alt_effect, 'temp_effect': temp_effect}


# =============================================================================
# SECTION 7: SUMMARY STATISTICS
# =============================================================================

def generate_summary_statistics(validation_metrics, optimal_spins, axis_sensitivity, env_effects):
    """Generate summary statistics for the paper."""
    print("\n" + "="*60)
    print("SECTION 7: Summary Statistics")
    print("="*60)

    summary = {
        'model_validation': {
            'R_squared': validation_metrics['R2'],
            'MAE_yards': validation_metrics['MAE'],
            'RMSE_yards': validation_metrics['RMSE'],
            'MAPE_percent': validation_metrics['MAPE']
        },
        'spin_optimization': {
            'optimal_spin_170mph': optimal_spins.get(170, 'N/A'),
            'optimal_spin_180mph': optimal_spins.get(180, 'N/A')
        },
        'spin_axis': {
            'sensitivity_yards_per_degree': axis_sensitivity
        },
        'environmental': {
            'altitude_effect_yards_per_5000ft': env_effects['altitude_effect'],
            'temperature_effect_yards_50F_range': env_effects['temp_effect']
        }
    }

    # Save to JSON
    import json
    json_path = os.path.join(DATA_DIR, 'research_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to: {json_path}")

    print(f"\n  KEY RESULTS:")
    print(f"    Model R² = {summary['model_validation']['R_squared']:.4f}")
    print(f"    Model MAE = {summary['model_validation']['MAE_yards']:.1f} yards")
    print(f"    Spin axis sensitivity = {summary['spin_axis']['sensitivity_yards_per_degree']:.2f} yards/degree")
    print(f"    Altitude effect = +{summary['environmental']['altitude_effect_yards_per_5000ft']:.1f} yards at 5000 ft")

    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete analysis pipeline."""
    print("\n" + "="*70)
    print("   MAGNUS EFFECT RESEARCH - COMPLETE ANALYSIS PIPELINE")
    print("="*70)

    # Run all analyses
    pga_data = create_pga_tour_data()
    validation_df, val_metrics = run_model_validation(pga_data)
    spin_df, optimal_spins = run_spin_rate_analysis()
    axis_df, axis_sensitivity = run_spin_axis_analysis()
    trajectories = create_trajectory_visualizations()
    alt_df, temp_df, env_effects = run_environmental_analysis()
    summary = generate_summary_statistics(val_metrics, optimal_spins, axis_sensitivity, env_effects)

    print("\n" + "="*70)
    print("   ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n  Data files saved to: {DATA_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")

    return {
        'validation': validation_df,
        'spin_analysis': spin_df,
        'axis_analysis': axis_df,
        'summary': summary
    }


if __name__ == "__main__":
    results = main()
