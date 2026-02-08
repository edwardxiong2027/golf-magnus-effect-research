# The Magnus Effect in Golf: A Computational Analysis of Spin-Induced Trajectory Deviations

## Research Overview

This research project investigates the **Magnus effect** in golf ball flight—the phenomenon where a spinning ball curves due to pressure differentials created by spin-air interaction. A physics-based computational model was developed to predict trajectory deviations and validated against PGA Tour TrackMan data.

**Target Journals:** Journal of Emerging Investigators (JEI) / Journal of High School Science (JHSS)

## Key Results

| Metric | Value |
|--------|-------|
| Model R² | 0.929 |
| Mean Absolute Error | 30.0 yards |
| Spin Axis Sensitivity | 1.95 yards/degree |
| Altitude Effect | +2.1 yards per 1,000 ft |

## Key Findings

1. **Spin Axis Effect:** Each degree of spin axis tilt produces ~2 yards of lateral deviation
2. **Optimal Backspin:** Maximum carry distance occurs at higher spin rates than PGA Tour averages
3. **Environmental Effects:** Playing at altitude (Denver) adds ~11 yards to driver distance

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
├── analysis/
│   └── 01_spin_rate_analysis.ipynb
└── RESEARCH_PROPOSAL.md
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

## Physics Model

The simulation implements:
- **Magnus Force:** Lift from ball spin interaction with air
- **Aerodynamic Drag:** Resistance reduced by dimple-induced turbulence
- **RK4 Integration:** 4th-order Runge-Kutta numerical solver
- **Environmental Corrections:** Altitude and temperature effects on air density

### Key Equations

**Spin Ratio:**
```
S = (ω × r) / v
```

**Lift Coefficient:**
```
C_L = 1.58 × S (capped at 0.28)
```

**Drag Coefficient:**
```
C_D = 0.255 + 0.13 × S
```

## Sample Output

```
Launch Conditions (PGA Tour Driver Average):
  Ball Speed: 171.0 mph
  Launch Angle: 10.4°
  Spin Rate: 2545 rpm

Simulation Results:
  Carry Distance: 275.7 yards
  Max Height: 29.8 yards
  Flight Time: 6.92 seconds
```

## Validation

Model validated against PGA Tour TrackMan averages for 12 clubs (Driver through PW):
- Strong correlation (R² = 0.93)
- Accurate predictions across all clubs
- Systematic effects correctly captured

## References

1. Bearman, P.W. & Harvey, J.K. (1976). "Golf ball aerodynamics." *Aeronautical Quarterly*.
2. Smits, A.J. & Smith, D.R. (1994). "A new aerodynamic model of a golf ball in flight."
3. Penner, A.R. (2003). "The physics of golf." *Reports on Progress in Physics*.

## Author

High School Junior | Physics & Golf Enthusiast

## License

MIT License - See LICENSE file for details
