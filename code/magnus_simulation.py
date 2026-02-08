"""
Golf Ball Trajectory Simulation with Magnus Effect

This module implements a physics-based model for simulating golf ball flight,
including the Magnus effect (spin-induced lift and side forces), drag, and gravity.

The model is calibrated against PGA Tour TrackMan data to ensure realistic predictions.

Author: High School Research Project
Date: February 2025

References:
    Bearman, P.W. & Harvey, J.K. (1976). Golf ball aerodynamics. Aeronautical Quarterly.
    Smits, A.J. & Smith, D.R. (1994). A new aerodynamic model of a golf ball in flight.
    Penner, A.R. (2003). The physics of golf. Reports on Progress in Physics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical Constants
GRAVITY = 9.81  # m/s²
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
GOLF_BALL_MASS = 0.04593  # kg (45.93 grams, USGA maximum)
GOLF_BALL_RADIUS = 0.02135  # m (42.7 mm diameter, USGA minimum)
GOLF_BALL_AREA = np.pi * GOLF_BALL_RADIUS**2  # Cross-sectional area


@dataclass
class LaunchConditions:
    """Initial launch conditions for a golf shot."""
    ball_speed: float  # m/s
    launch_angle: float  # degrees (vertical angle from ground)
    launch_direction: float  # degrees (horizontal angle, 0 = straight)
    spin_rate: float  # rpm (revolutions per minute)
    spin_axis: float  # degrees (0 = pure backspin, +ve = fade/slice, -ve = draw/hook)
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

    This model implements:
    - Magnus effect: Lift and side force from ball spin
    - Aerodynamic drag: Resistance from air, reduced by dimples
    - Gravitational acceleration

    The aerodynamic coefficients are calibrated to match PGA Tour
    TrackMan data, providing accurate trajectory predictions.
    """

    def __init__(self):
        self.trajectory: List[TrajectoryPoint] = []

    def get_air_density(self, altitude: float, temperature: float) -> float:
        """
        Calculate air density adjusted for altitude and temperature.

        Uses the barometric formula and ideal gas law.

        Args:
            altitude: Height above sea level in meters
            temperature: Air temperature in Celsius

        Returns:
            Air density in kg/m³
        """
        # Temperature in Kelvin
        T = temperature + 273.15
        T_standard = 288.15  # Standard temperature (15°C in Kelvin)

        # Pressure ratio using barometric formula
        # P/P_0 = (1 - Lh/T_0)^(gM/RL) where L = lapse rate
        if altitude < 11000:  # Troposphere
            pressure_ratio = (1 - 0.0065 * altitude / T_standard) ** 5.2561
        else:
            pressure_ratio = 0.22336  # Approximate for stratosphere

        # Density from ideal gas law: ρ = PM/(RT)
        # ρ/ρ_0 = (P/P_0) × (T_0/T)
        rho = AIR_DENSITY_SEA_LEVEL * pressure_ratio * (T_standard / T)

        return rho

    def get_aerodynamic_coefficients(self, spin_rate_rpm: float, ball_speed: float) -> tuple:
        """
        Calculate lift and drag coefficients for a spinning golf ball.

        This function uses empirically calibrated relationships based on
        wind tunnel studies and trajectory data from PGA Tour.

        The key physics:
        - Spin creates pressure differential via Magnus effect
        - Dimples create turbulent boundary layer, reducing drag
        - Lift coefficient increases with spin ratio
        - Drag coefficient is relatively constant for golf ball Re numbers

        Args:
            spin_rate_rpm: Ball spin rate in RPM
            ball_speed: Ball speed in m/s

        Returns:
            Tuple of (C_L, C_D) - lift and drag coefficients
        """
        # Convert spin to rad/s
        omega = spin_rate_rpm * 2 * np.pi / 60

        # Spin ratio: ratio of surface speed to ball speed
        # S = (ω × r) / v
        if ball_speed > 1.0:
            spin_ratio = (omega * GOLF_BALL_RADIUS) / ball_speed
        else:
            spin_ratio = 0

        # Lift coefficient (Magnus effect)
        # Calibrated to match PGA Tour TrackMan data
        # The Magnus force provides lift proportional to spin × velocity
        # Reference: Bearman & Harvey (1976), Smits & Smith (1994)
        # Final calibration based on PGA Tour driver data (275 yards, 32 yards apex)
        C_L = min(1.58 * spin_ratio, 0.28)

        # Drag coefficient
        # Dimpled golf ball has C_D ≈ 0.24-0.28 in typical flight
        # Dimples create turbulent boundary layer, delaying separation
        # Final calibration for PGA Tour trajectory matching
        C_D = 0.255 + 0.13 * spin_ratio

        return C_L, C_D

    def compute_acceleration(self, position: np.ndarray, velocity: np.ndarray,
                            conditions: LaunchConditions, rho: float) -> np.ndarray:
        """
        Compute acceleration on the golf ball from all forces.

        Forces considered:
        1. Gravity: F_g = -mg (downward)
        2. Drag: F_D = -½ρv²C_D A (opposing velocity)
        3. Magnus Lift: F_L = ½ρv²C_L A (perpendicular to velocity, upward for backspin)
        4. Magnus Side: F_S = ½ρv²C_L A × sin(spin_axis) (lateral, for tilted spin axis)

        Args:
            position: [x, y, z] position in meters
            velocity: [vx, vy, vz] velocity in m/s
            conditions: Launch conditions (for spin information)
            rho: Air density in kg/m³

        Returns:
            [ax, ay, az] acceleration in m/s²
        """
        vx, vy, vz = velocity
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        if v < 0.5:  # Ball essentially stopped
            return np.array([0.0, -GRAVITY, 0.0])

        # Unit velocity vector
        v_hat = velocity / v

        # Get aerodynamic coefficients
        C_L, C_D = self.get_aerodynamic_coefficients(conditions.spin_rate, v)

        # Dynamic pressure: q = ½ρv²
        q = 0.5 * rho * v**2

        # ===== DRAG FORCE =====
        # Opposes motion: F_D = -q × A × C_D × v_hat
        F_drag = -q * GOLF_BALL_AREA * C_D * v_hat

        # ===== MAGNUS FORCE (LIFT AND SIDE FORCE) =====
        # For a golf ball with spin:
        # - Backspin (spin_axis = 0): Creates upward lift
        # - Tilted spin axis: Creates both lift and side force

        # Spin axis angle: 0 = pure backspin, ±90 = pure sidespin
        axis_rad = np.radians(conditions.spin_axis)

        # Horizontal velocity direction (for reference frame)
        v_horizontal = np.sqrt(vx**2 + vz**2)
        if v_horizontal > 0.1:
            # Forward direction (normalized horizontal velocity)
            forward = np.array([vx, 0, vz]) / v_horizontal
            # Right direction (perpendicular to forward, in horizontal plane)
            right = np.array([forward[2], 0, -forward[0]])
            # Up direction
            up = np.array([0, 1, 0])
        else:
            forward = np.array([0, 0, 1])
            right = np.array([1, 0, 0])
            up = np.array([0, 1, 0])

        # Lift force magnitude (Magnus effect)
        F_L_magnitude = q * GOLF_BALL_AREA * C_L

        # Decompose into vertical lift and lateral force based on spin axis
        # Pure backspin (axis=0): all lift is vertical
        # Tilted axis: some lift becomes lateral
        lift_vertical = F_L_magnitude * np.cos(axis_rad)  # Upward lift
        lift_lateral = F_L_magnitude * np.sin(axis_rad)   # Side force (+ = right)

        # Magnus force vector
        F_magnus = lift_vertical * up + lift_lateral * right

        # ===== GRAVITY =====
        F_gravity = np.array([0, -GOLF_BALL_MASS * GRAVITY, 0])

        # ===== TOTAL FORCE AND ACCELERATION =====
        F_total = F_drag + F_magnus + F_gravity
        acceleration = F_total / GOLF_BALL_MASS

        return acceleration

    def simulate(self, conditions: LaunchConditions, dt: float = 0.001,
                 max_time: float = 15.0) -> List[TrajectoryPoint]:
        """
        Simulate golf ball trajectory using 4th-order Runge-Kutta integration.

        The RK4 method provides excellent accuracy for trajectory simulation,
        with local error O(dt^5) and global error O(dt^4).

        Args:
            conditions: Initial launch conditions
            dt: Time step in seconds (default: 1ms for accuracy)
            max_time: Maximum simulation time in seconds

        Returns:
            List of trajectory points
        """
        # Get air density for conditions
        rho = self.get_air_density(conditions.altitude, conditions.temperature)

        # Initial velocity from launch conditions
        v0 = conditions.ball_speed
        theta = np.radians(conditions.launch_angle)
        phi = np.radians(conditions.launch_direction)

        # Decompose initial velocity
        vy0 = v0 * np.sin(theta)  # Vertical component
        v_horiz = v0 * np.cos(theta)  # Horizontal component
        vz0 = v_horiz * np.cos(phi)  # Downrange
        vx0 = v_horiz * np.sin(phi)  # Lateral

        # Initial state
        position = np.array([0.0, 0.0, 0.0])  # [x, y, z]
        velocity = np.array([vx0, vy0, vz0])

        # Clear trajectory
        self.trajectory = []

        # Time integration using RK4
        t = 0.0
        while t < max_time:
            # Store current point
            self.trajectory.append(TrajectoryPoint(
                time=t,
                x=position[0], y=position[1], z=position[2],
                vx=velocity[0], vy=velocity[1], vz=velocity[2]
            ))

            # Stop if ball hits ground (after initial launch)
            if t > 0.1 and position[1] <= 0:
                break

            # RK4 integration for velocity (acceleration derivative)
            # State: position and velocity
            k1_v = self.compute_acceleration(position, velocity, conditions, rho)
            k1_p = velocity

            k2_v = self.compute_acceleration(position + 0.5*dt*k1_p,
                                             velocity + 0.5*dt*k1_v, conditions, rho)
            k2_p = velocity + 0.5*dt*k1_v

            k3_v = self.compute_acceleration(position + 0.5*dt*k2_p,
                                             velocity + 0.5*dt*k2_v, conditions, rho)
            k3_p = velocity + 0.5*dt*k2_v

            k4_v = self.compute_acceleration(position + dt*k3_p,
                                             velocity + dt*k3_v, conditions, rho)
            k4_p = velocity + dt*k3_v

            # Update state
            velocity = velocity + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            position = position + (dt/6) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

            t += dt

        return self.trajectory

    def get_carry_distance(self) -> float:
        """Get carry distance in meters (downrange position at landing)."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].z

    def get_lateral_deviation(self) -> float:
        """Get lateral deviation in meters (positive = right of target)."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].x

    def get_max_height(self) -> float:
        """Get apex height in meters."""
        if not self.trajectory:
            return 0.0
        return max(p.y for p in self.trajectory)

    def get_flight_time(self) -> float:
        """Get total flight time in seconds."""
        if not self.trajectory:
            return 0.0
        return self.trajectory[-1].time

    def get_land_angle(self) -> float:
        """Get landing angle in degrees (descent angle at impact)."""
        if len(self.trajectory) < 2:
            return 0.0
        # Use last few points to estimate descent angle
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]
        dz = p2.z - p1.z
        dy = p2.y - p1.y
        if dz > 0:
            return np.degrees(np.arctan(-dy / dz))
        return 45.0  # Default if calculation fails


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def meters_to_yards(m: float) -> float:
    """Convert meters to yards."""
    return m * 1.09361


def yards_to_meters(yards: float) -> float:
    """Convert yards to meters."""
    return yards / 1.09361


def mph_to_ms(mph: float) -> float:
    """Convert miles per hour to meters per second."""
    return mph * 0.44704


def ms_to_mph(ms: float) -> float:
    """Convert meters per second to miles per hour."""
    return ms / 0.44704


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_trajectory_3d(simulator: GolfBallSimulator, title: str = "Golf Ball Trajectory"):
    """Create 3D visualization of ball trajectory."""
    if not simulator.trajectory:
        print("No trajectory data to plot")
        return None

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = [meters_to_yards(p.x) for p in simulator.trajectory]
    y = [meters_to_yards(p.y) for p in simulator.trajectory]
    z = [meters_to_yards(p.z) for p in simulator.trajectory]

    ax.plot(z, x, y, 'b-', linewidth=2, label='Trajectory')
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
    """Create 2D side-view and top-view plots of trajectory."""
    if not simulator.trajectory:
        print("No trajectory data to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    z = [meters_to_yards(p.z) for p in simulator.trajectory]
    y = [meters_to_yards(p.y) for p in simulator.trajectory]
    x = [meters_to_yards(p.x) for p in simulator.trajectory]

    # Side view (distance vs height)
    ax1.plot(z, y, 'b-', linewidth=2)
    ax1.scatter([z[0]], [y[0]], color='green', s=100, zorder=5, label='Launch')
    ax1.scatter([z[-1]], [y[-1]], color='red', s=100, zorder=5, label='Landing')
    ax1.set_xlabel('Distance (yards)')
    ax1.set_ylabel('Height (yards)')
    ax1.set_title('Side View')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.legend()

    # Top view (distance vs lateral)
    ax2.plot(z, x, 'b-', linewidth=2)
    ax2.scatter([z[0]], [x[0]], color='green', s=100, zorder=5, label='Launch')
    ax2.scatter([z[-1]], [x[-1]], color='red', s=100, zorder=5, label='Landing')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance (yards)')
    ax2.set_ylabel('Lateral Deviation (yards)')
    ax2.set_title('Top View')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN - DEMONSTRATION AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  GOLF BALL TRAJECTORY SIMULATION - MAGNUS EFFECT MODEL")
    print("=" * 70)

    sim = GolfBallSimulator()

    # Test with typical PGA Tour driver conditions
    conditions = LaunchConditions(
        ball_speed=mph_to_ms(171),  # PGA Tour average
        launch_angle=10.4,
        launch_direction=0,
        spin_rate=2545,
        spin_axis=0,  # Pure backspin
        altitude=0,
        temperature=20
    )

    print(f"\nLaunch Conditions (PGA Tour Driver Average):")
    print(f"  Ball Speed: {ms_to_mph(conditions.ball_speed):.1f} mph")
    print(f"  Launch Angle: {conditions.launch_angle}°")
    print(f"  Spin Rate: {conditions.spin_rate} rpm")
    print(f"  Spin Axis: {conditions.spin_axis}°")

    # Run simulation
    sim.simulate(conditions)

    print(f"\nSimulation Results:")
    print(f"  Carry Distance: {meters_to_yards(sim.get_carry_distance()):.1f} yards (Target: 275)")
    print(f"  Max Height: {meters_to_yards(sim.get_max_height()):.1f} yards (Target: 32)")
    print(f"  Flight Time: {sim.get_flight_time():.2f} seconds")
    print(f"  Land Angle: {sim.get_land_angle():.1f}° (Target: 38)")

    # Test spin axis effect
    print("\n" + "-" * 50)
    print("Spin Axis Effect (Draw/Fade):")
    print("-" * 50)

    for axis in [-10, -5, 0, 5, 10]:
        conditions.spin_axis = axis
        sim.simulate(conditions)
        deviation = meters_to_yards(sim.get_lateral_deviation())
        carry = meters_to_yards(sim.get_carry_distance())
        label = "DRAW" if axis < 0 else "FADE" if axis > 0 else "STRAIGHT"
        print(f"  Axis {axis:+3d}°: {deviation:+6.1f} yards lateral, {carry:.1f} yards carry ({label})")
