"""
Golf Ball Trajectory Simulation with Magnus Effect

This module implements a physics-based model for simulating golf ball flight,
including the Magnus effect (spin-induced lift and side forces), drag, and gravity.

Author: High School Research Project
Date: February 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical Constants
GRAVITY = 9.81  # m/s²
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
GOLF_BALL_MASS = 0.04593  # kg (45.93 grams)
GOLF_BALL_RADIUS = 0.02135  # m (42.7 mm diameter)
GOLF_BALL_AREA = np.pi * GOLF_BALL_RADIUS**2  # Cross-sectional area


@dataclass
class LaunchConditions:
    """Initial launch conditions for a golf shot."""
    ball_speed: float  # m/s
    launch_angle: float  # degrees (vertical angle from ground)
    launch_direction: float  # degrees (horizontal angle, 0 = straight)
    spin_rate: float  # rpm
    spin_axis: float  # degrees (0 = pure backspin, +90 = pure sidespin right)
    altitude: float = 0.0  # m above sea level
    temperature: float = 20.0  # Celsius


@dataclass
class TrajectoryPoint:
    """A single point in the ball's trajectory."""
    time: float  # seconds
    x: float  # lateral position (m), positive = right
    y: float  # vertical position (m)
    z: float  # downrange position (m)
    vx: float  # lateral velocity (m/s)
    vy: float  # vertical velocity (m/s)
    vz: float  # downrange velocity (m/s)


class GolfBallSimulator:
    """
    Simulates golf ball flight using physics-based modeling.

    Implements:
    - Magnus effect (lift and side force from spin)
    - Aerodynamic drag
    - Gravitational acceleration
    - Altitude and temperature corrections
    """

    def __init__(self):
        self.trajectory: List[TrajectoryPoint] = []

    def get_air_density(self, altitude: float, temperature: float) -> float:
        """
        Calculate air density adjusted for altitude and temperature.

        Uses barometric formula and ideal gas approximation.

        Args:
            altitude: Height above sea level in meters
            temperature: Air temperature in Celsius

        Returns:
            Air density in kg/m³
        """
        # Temperature in Kelvin
        T = temperature + 273.15
        T_standard = 288.15  # Standard temperature (15°C)

        # Pressure ratio (barometric formula)
        pressure_ratio = (1 - 0.0065 * altitude / T_standard) ** 5.2561

        # Density adjusted for temperature and altitude
        rho = AIR_DENSITY_SEA_LEVEL * pressure_ratio * (T_standard / T)

        return rho

    def get_spin_parameter(self, spin_rate_rpm: float, ball_speed: float) -> float:
        """
        Calculate dimensionless spin parameter.

        S = (ω × r) / v

        Args:
            spin_rate_rpm: Spin rate in revolutions per minute
            ball_speed: Ball speed in m/s

        Returns:
            Dimensionless spin parameter
        """
        omega = spin_rate_rpm * 2 * np.pi / 60  # Convert to rad/s
        return (omega * GOLF_BALL_RADIUS) / ball_speed if ball_speed > 0 else 0

    def get_lift_coefficient(self, spin_parameter: float) -> float:
        """
        Estimate lift coefficient based on spin parameter.

        Uses empirical relationship from golf ball aerodynamics research.

        Args:
            spin_parameter: Dimensionless spin parameter

        Returns:
            Lift coefficient C_L
        """
        # Empirical fit based on Bearman & Harvey (1976) and subsequent studies
        # C_L increases with spin parameter but saturates at high spin
        C_L = 0.54 * spin_parameter ** 0.4
        return min(C_L, 0.35)  # Cap at realistic maximum

    def get_drag_coefficient(self, spin_parameter: float, reynolds_number: float) -> float:
        """
        Estimate drag coefficient for a dimpled golf ball.

        Args:
            spin_parameter: Dimensionless spin parameter
            reynolds_number: Reynolds number of the flow

        Returns:
            Drag coefficient C_D
        """
        # Base drag for dimpled ball (lower than smooth sphere)
        C_D_base = 0.25

        # Drag increases slightly with spin (Magnus effect adds induced drag)
        C_D = C_D_base + 0.15 * spin_parameter

        return min(C_D, 0.45)  # Cap at realistic maximum

    def compute_forces(self, state: np.ndarray, conditions: LaunchConditions,
                       rho: float) -> np.ndarray:
        """
        Compute all forces acting on the golf ball.

        Args:
            state: [x, y, z, vx, vy, vz] position and velocity
            conditions: Launch conditions (for spin info)
            rho: Air density

        Returns:
            [ax, ay, az] accelerations in m/s²
        """
        x, y, z, vx, vy, vz = state

        # Velocity magnitude
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v < 0.1:  # Ball essentially stopped
            return np.array([0, -GRAVITY, 0, 0, 0, 0])

        # Unit velocity vector
        v_hat = np.array([vx, vy, vz]) / v

        # Spin parameter
        spin_param = self.get_spin_parameter(conditions.spin_rate, v)

        # Aerodynamic coefficients
        C_L = self.get_lift_coefficient(spin_param)
        Re = rho * v * 2 * GOLF_BALL_RADIUS / 1.81e-5  # Reynolds number
        C_D = self.get_drag_coefficient(spin_param, Re)

        # Dynamic pressure
        q = 0.5 * rho * v**2

        # Drag force (opposes velocity)
        F_drag = -q * GOLF_BALL_AREA * C_D * v_hat

        # Spin axis in 3D (spin_axis=0 is pure backspin)
        spin_axis_rad = np.radians(conditions.spin_axis)

        # Spin vector direction (perpendicular to velocity and lift direction)
        # For backspin, spin axis is horizontal and perpendicular to flight
        omega_hat = np.array([
            np.sin(spin_axis_rad),  # Side component
            0,
            -np.cos(spin_axis_rad)  # Forward component
        ])

        # Magnus force direction: ω × v (cross product)
        magnus_dir = np.cross(omega_hat, v_hat)
        magnus_dir_norm = np.linalg.norm(magnus_dir)
        if magnus_dir_norm > 0:
            magnus_dir = magnus_dir / magnus_dir_norm

        # Magnus force magnitude
        F_magnus = q * GOLF_BALL_AREA * C_L * magnus_dir

        # Gravity
        F_gravity = np.array([0, -GOLF_BALL_MASS * GRAVITY, 0])

        # Total force
        F_total = F_drag + F_magnus + F_gravity

        # Acceleration
        a = F_total / GOLF_BALL_MASS

        return np.array([vx, vy, vz, a[0], a[1], a[2]])

    def simulate(self, conditions: LaunchConditions, dt: float = 0.001,
                 max_time: float = 15.0) -> List[TrajectoryPoint]:
        """
        Simulate golf ball trajectory using RK4 integration.

        Args:
            conditions: Initial launch conditions
            dt: Time step in seconds
            max_time: Maximum simulation time in seconds

        Returns:
            List of trajectory points
        """
        # Get air density
        rho = self.get_air_density(conditions.altitude, conditions.temperature)

        # Convert launch conditions to initial state
        v0 = conditions.ball_speed
        theta = np.radians(conditions.launch_angle)
        phi = np.radians(conditions.launch_direction)

        # Initial velocity components
        vy0 = v0 * np.sin(theta)
        v_horizontal = v0 * np.cos(theta)
        vz0 = v_horizontal * np.cos(phi)
        vx0 = v_horizontal * np.sin(phi)

        # Initial state: [x, y, z, vx, vy, vz]
        state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])

        self.trajectory = []
        t = 0.0

        while t < max_time and state[1] >= 0:  # Stop when ball hits ground
            # Store trajectory point
            self.trajectory.append(TrajectoryPoint(
                time=t,
                x=state[0], y=state[1], z=state[2],
                vx=state[3], vy=state[4], vz=state[5]
            ))

            # RK4 integration
            k1 = self.compute_forces(state, conditions, rho)
            k2 = self.compute_forces(state + 0.5*dt*k1, conditions, rho)
            k3 = self.compute_forces(state + 0.5*dt*k2, conditions, rho)
            k4 = self.compute_forces(state + dt*k3, conditions, rho)

            state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt

        return self.trajectory

    def get_carry_distance(self) -> float:
        """Get carry distance in meters."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].z

    def get_lateral_deviation(self) -> float:
        """Get lateral deviation in meters (positive = right)."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].x

    def get_max_height(self) -> float:
        """Get maximum height in meters."""
        if not self.trajectory:
            return 0.0
        return max(p.y for p in self.trajectory)

    def get_flight_time(self) -> float:
        """Get total flight time in seconds."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].time


def meters_to_yards(m: float) -> float:
    """Convert meters to yards."""
    return m * 1.09361


def mph_to_ms(mph: float) -> float:
    """Convert miles per hour to meters per second."""
    return mph * 0.44704


def plot_trajectory_3d(simulator: GolfBallSimulator, title: str = "Golf Ball Trajectory"):
    """Create 3D plot of trajectory."""
    if not simulator.trajectory:
        print("No trajectory data to plot")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = [meters_to_yards(p.x) for p in simulator.trajectory]
    y = [meters_to_yards(p.y) for p in simulator.trajectory]
    z = [meters_to_yards(p.z) for p in simulator.trajectory]

    ax.plot(z, x, y, 'b-', linewidth=2)
    ax.scatter([z[0]], [x[0]], [y[0]], color='green', s=100, label='Launch')
    ax.scatter([z[-1]], [x[-1]], [y[-1]], color='red', s=100, label='Landing')

    ax.set_xlabel('Distance (yards)')
    ax.set_ylabel('Lateral (yards)')
    ax.set_zlabel('Height (yards)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_trajectory_2d(simulator: GolfBallSimulator, title: str = "Golf Ball Trajectory"):
    """Create 2D plots of trajectory (side view and top view)."""
    if not simulator.trajectory:
        print("No trajectory data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    z = [meters_to_yards(p.z) for p in simulator.trajectory]
    y = [meters_to_yards(p.y) for p in simulator.trajectory]
    x = [meters_to_yards(p.x) for p in simulator.trajectory]

    # Side view
    ax1.plot(z, y, 'b-', linewidth=2)
    ax1.scatter([z[0]], [y[0]], color='green', s=100, zorder=5)
    ax1.scatter([z[-1]], [y[-1]], color='red', s=100, zorder=5)
    ax1.set_xlabel('Distance (yards)')
    ax1.set_ylabel('Height (yards)')
    ax1.set_title('Side View')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Top view
    ax2.plot(z, x, 'b-', linewidth=2)
    ax2.scatter([z[0]], [x[0]], color='green', s=100, zorder=5, label='Launch')
    ax2.scatter([z[-1]], [x[-1]], color='red', s=100, zorder=5, label='Landing')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance (yards)')
    ax2.set_ylabel('Lateral Deviation (yards)')
    ax2.set_title('Top View (Bird\'s Eye)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# Example usage and demonstration
if __name__ == "__main__":
    # Create simulator
    sim = GolfBallSimulator()

    # Typical PGA Tour driver conditions
    conditions = LaunchConditions(
        ball_speed=mph_to_ms(170),  # 170 mph ball speed
        launch_angle=10.5,  # 10.5 degrees
        launch_direction=0,  # Straight
        spin_rate=2500,  # 2500 rpm backspin
        spin_axis=0,  # Pure backspin (no sidespin)
        altitude=0,
        temperature=20
    )

    print("=" * 60)
    print("Golf Ball Trajectory Simulation - Magnus Effect Model")
    print("=" * 60)
    print(f"\nLaunch Conditions:")
    print(f"  Ball Speed: {conditions.ball_speed / 0.44704:.1f} mph")
    print(f"  Launch Angle: {conditions.launch_angle}°")
    print(f"  Spin Rate: {conditions.spin_rate} rpm")
    print(f"  Spin Axis: {conditions.spin_axis}° (0 = pure backspin)")

    # Run simulation
    trajectory = sim.simulate(conditions)

    print(f"\nResults:")
    print(f"  Carry Distance: {meters_to_yards(sim.get_carry_distance()):.1f} yards")
    print(f"  Max Height: {meters_to_yards(sim.get_max_height()):.1f} yards")
    print(f"  Lateral Deviation: {meters_to_yards(sim.get_lateral_deviation()):.1f} yards")
    print(f"  Flight Time: {sim.get_flight_time():.2f} seconds")

    # Demonstrate spin axis effect (slice/hook)
    print("\n" + "=" * 60)
    print("Spin Axis Effect Demonstration")
    print("=" * 60)

    for spin_axis in [-10, 0, 10]:
        conditions.spin_axis = spin_axis
        sim.simulate(conditions)

        deviation = meters_to_yards(sim.get_lateral_deviation())
        direction = "LEFT" if deviation < 0 else "RIGHT" if deviation > 0 else "STRAIGHT"

        print(f"  Spin Axis {spin_axis:+3d}°: {abs(deviation):.1f} yards {direction}")

    # Create plots
    conditions.spin_axis = 5  # Slight fade
    sim.simulate(conditions)
    plot_trajectory_2d(sim, "Driver Shot with 5° Fade (Spin Axis = +5°)")
    plt.savefig('../figures/trajectory_example.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to figures/trajectory_example.png")
    plt.show()
