"""Kinematic closure solver for the modular 1-DoF / 3-joint finger linkage."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from .finger_config import FingerExtendedGeometry
except ImportError:
    from finger_config import FingerExtendedGeometry  # type: ignore


Array = np.ndarray


def rotation(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def angle_of(v: Array) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def circle_intersections(c0: Array, r0: float, c1: Array, r1: float, eps: float = 1e-12) -> Array:
    """
    Return the two intersections of circles:
        ||x - c0|| = r0
        ||x - c1|| = r1
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    d_vec = c1 - c0
    d = float(np.linalg.norm(d_vec))

    if d < eps:
        raise ValueError("Circle centers are coincident; closure branch is under-defined")
    if d > r0 + r1 + eps:
        raise ValueError(f"No circle intersection: centers too far apart, d={d}, r0+r1={r0 + r1}")
    if d < abs(r0 - r1) - eps:
        raise ValueError(f"No circle intersection: one circle contains the other, d={d}, |r0-r1|={abs(r0 - r1)}")

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = r0 * r0 - a * a
    h = math.sqrt(max(0.0, h_sq))

    e = d_vec / d
    p = c0 + a * e
    n = np.array([-e[1], e[0]], dtype=float)
    return np.vstack((p + h * n, p - h * n))


def choose_branch(candidates: Array, previous_point: Array | None, reference_point: Array) -> Array:
    """Choose a circle-intersection branch by continuity, then by reference pose."""
    target = reference_point if previous_point is None else previous_point
    distances = np.linalg.norm(candidates - target[None, :], axis=1)
    return candidates[int(np.argmin(distances))]


@dataclass(frozen=True)
class FingerState:
    """Solved finger pose for one input angle."""

    q1: float
    q2: float
    q3: float

    O: Array
    O_prime: Array
    A: Array
    A_prime: Array
    B: Array
    B_prime: Array
    C_prime: Array
    D: Array

    def point_dict(self) -> dict[str, Array]:
        return {
            "O": self.O,
            "O_prime": self.O_prime,
            "A": self.A,
            "A_prime": self.A_prime,
            "B": self.B,
            "B_prime": self.B_prime,
            "C_prime": self.C_prime,
            "D": self.D,
        }

    def closure_residuals(self, geometry: FingerExtendedGeometry) -> dict[str, float]:
        return {
            "AB_prime": float(np.linalg.norm(self.B_prime - self.A) - geometry.AB_prime_length),
            "O_prime_B_prime": float(
                np.linalg.norm(self.B_prime - self.O_prime) - geometry.O_prime_B_prime_length
            ),
            "BC_prime": float(np.linalg.norm(self.C_prime - self.B) - geometry.BC_prime_length),
            "A_prime_C_prime": float(
                np.linalg.norm(self.C_prime - self.A_prime) - geometry.A_prime_C_prime_length
            ),
        }

    def max_abs_closure_residual(self, geometry: FingerExtendedGeometry) -> float:
        residuals = self.closure_residuals(geometry)
        return max(abs(value) for value in residuals.values())


@dataclass(frozen=True)
class FingerTrajectory:
    """A sequence of solved finger states."""

    states: tuple[FingerState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))

    def closure_residual_array(self, geometry: FingerExtendedGeometry) -> Array:
        names = ("AB_prime", "O_prime_B_prime", "BC_prime", "A_prime_C_prime")
        return np.array([[state.closure_residuals(geometry)[name] for name in names] for state in self.states])

    def max_abs_closure_residual(self, geometry: FingerExtendedGeometry) -> float:
        if not self.states:
            return 0.0
        return max(state.max_abs_closure_residual(geometry) for state in self.states)


class FingerKinematicModel:
    """
    Circle-intersection kinematic closure solver.

    Linkage lengths are hard closure constraints, not objective penalties:
        ||B_prime - A||       = geometry.AB_prime_length
        ||B_prime - O_prime|| = geometry.O_prime_B_prime_length
        ||C_prime - B||       = geometry.BC_prime_length
        ||C_prime - A_prime|| = geometry.A_prime_C_prime_length
    """

    def __init__(
        self,
        geometry: FingerExtendedGeometry,
        closure_tol: float = 1e-8,
        validate_closure: bool = True,
    ) -> None:
        if closure_tol < 0.0:
            raise ValueError(f"closure_tol must be nonnegative, got {closure_tol}")
        self.geometry = geometry
        self.closure_tol = closure_tol
        self.validate_closure = validate_closure

    def solve(self, q1: float, previous_state: FingerState | None = None) -> FingerState:
        g = self.geometry
        R1 = rotation(q1)

        O = g.O
        O_prime = g.O_prime
        A = O + R1 @ g.fixed.OA_vec
        A_prime = A + R1 @ g.design.AA_prime_vec

        previous_B_prime = None if previous_state is None else previous_state.B_prime
        B_prime_candidates = circle_intersections(
            A,
            g.AB_prime_length,
            O_prime,
            g.O_prime_B_prime_length,
        )
        B_prime = choose_branch(B_prime_candidates, previous_B_prime, g.B_prime)

        q2 = angle_of(B_prime - A) - angle_of(g.design.AB_prime_vec)
        R2 = rotation(q2)
        B = A + R2 @ g.fixed.AB_vec

        previous_C_prime = None if previous_state is None else previous_state.C_prime
        C_prime_candidates = circle_intersections(
            B,
            g.BC_prime_length,
            A_prime,
            g.A_prime_C_prime_length,
        )
        C_prime = choose_branch(C_prime_candidates, previous_C_prime, g.C_prime)

        q3 = angle_of(C_prime - B) - angle_of(g.design.BC_prime_vec)
        R3 = rotation(q3)
        D = B + R3 @ g.fixed.BD_vec

        state = FingerState(
            q1=q1,
            q2=q2,
            q3=q3,
            O=O,
            O_prime=O_prime,
            A=A,
            A_prime=A_prime,
            B=B,
            B_prime=B_prime,
            C_prime=C_prime,
            D=D,
        )
        if self.validate_closure:
            self.check_closure(state)
        return state

    def solve_trajectory(self, q1_values: Array) -> FingerTrajectory:
        states: list[FingerState] = []
        previous_state: FingerState | None = None
        for q1 in np.asarray(q1_values, dtype=float).reshape(-1):
            state = self.solve(float(q1), previous_state)
            states.append(state)
            previous_state = state
        return FingerTrajectory(tuple(states))

    def check_closure(self, state: FingerState) -> None:
        residuals = state.closure_residuals(self.geometry)
        max_residual = max(abs(value) for value in residuals.values())
        if max_residual > self.closure_tol:
            raise RuntimeError(
                f"Kinematic closure residual too large: {max_residual}, residuals={residuals}"
            )
