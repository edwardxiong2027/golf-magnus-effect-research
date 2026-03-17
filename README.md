# Spin Axis Tilt Linearly Predicts Lateral Deviation in Golf Ball Flight via the Magnus Effect

## Research Overview

This research uses a validated computational physics simulation to test three hypotheses about how the **Magnus effect** governs golf ball trajectory. Rather than presenting the simulation as an end in itself, we use it as an experimental tool to answer specific scientific questions about the relationships between spin parameters, environmental conditions, and trajectory outcomes.

**Format:** Journal of Emerging Investigators (JEI) — Hypothesis-Driven Research

## Hypotheses & Key Results

### Hypothesis 1: Lateral deviation increases linearly with spin axis tilt
- **Result:** Strongly supported (R² = 0.998)
- **Finding:** 1.95 yards lateral deviation per degree of spin axis tilt

### Hypothesis 2: Optimal backspin rate increases with ball speed
- **Result:** Supported
- **Finding:** Optimal backspin increased from ~4,200 rpm (150 mph) to ~5,400 rpm (180 mph)

### Hypothesis 3: Altitude affects carry distance more than temperature
- **Result:** Supported
- **Finding:** Altitude produces 6.8× greater carry distance change per unit air density variation

### Simulation Validation
| Metric | Value |
|--------|-------|
| R² (12 club types) | 0.929 |
| Mean Absolute Error | 30.0 yards |
| Validated against | PGA Tour TrackMan 2023-2024 |

## Project Structure

```
golf-magnus-effect-research/
├── code/
│   ├── magnus_simulation.py     # Core physics simulation
│   └── run_full_analysis.py     # Complete analysis pipeline
├── data/
│   ├── pga_tour_trackman_averages.csv
│   ├── model_validation_results.csv
│   ├── spin_rate_analysis.csv
│   ├── spin_axis_analysis.csv
│   └── research_summary.json
├── figures/
│   ├── fig1_model_validation.png
│   ├── fig2_spin_rate_analysis.png
│   ├── fig3_spin_axis_analysis.png
│   ├── fig4_trajectory_comparison.png
│   ├── fig5_3d_trajectory.png
│   └── fig6_environmental_effects.png
├── paper/
│   └── magnus_effect_golf_paper.md
├── webapp/
│   └── public/index.html         # Interactive web application
├── analysis/
│   └── 01_spin_rate_analysis.ipynb
├── RESEARCH_PROPOSAL.md
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas seaborn

# Run complete analysis
python code/run_full_analysis.py

# Run individual simulation
python code/magnus_simulation.py
```

## Interactive Web Application

An interactive webapp allows exploration of the simulation results:

```bash
cd webapp
firebase deploy
```

## Physics Model

The simulation implements:
- **Magnus Force:** Lift from ball spin interaction with air
- **Aerodynamic Drag:** Resistance reduced by dimple-induced turbulence
- **RK4 Integration:** 4th-order Runge-Kutta numerical solver
- **Environmental Corrections:** Altitude and temperature effects on air density

## References

1. Magnus, H.G. (1852). "On the deviation of projectiles." *Annalen der Physik*.
2. Bearman, P.W. & Harvey, J.K. (1976). "Golf ball aerodynamics." *Aeronautical Quarterly*.
3. Penner, A.R. (2003). "The physics of golf." *Reports on Progress in Physics*.

## Author

Edward Xiong | Diamond Bar High School

## License

MIT License - See LICENSE file for details
