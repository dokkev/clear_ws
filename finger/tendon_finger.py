"""Underactuated tendon-driven serial finger with torsional joint springs.

The deterministic linkage model in :mod:`finger.finger_kinematics` prescribes
all joint angles from one input by hard kinematic closure. This model is the
comparison case: one tendon command pulls a serial 3R finger, while torsional
springs and contact locks determine how the command is redistributed.

Sign convention:
    command >= 0 produces flexion, represented by negative joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray


def _as_array3(values: Sequence[float], name: str) -> Array:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {arr.size}")
    return arr


def rotation(theta: float) -> Array:
    """Return a 2D rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass(frozen=True)
class TendonFingerConfig:
    """Parameters for a 3-joint tendon-driven underactuated finger.

    ``spring_stiffness`` is the torsional stiffness at MCP/PIP/DIP. A smaller
    stiffness lets that joint flex more for the same tendon command.

    ``tendon_moment_arms`` controls how strongly the single tendon command
    pulls each joint. With the default sign convention, larger positive moment
    arms create larger negative flexion angles.
    """

    link_lengths: tuple[float, float, float] = (40.0, 28.0, 22.0)
    spring_stiffness: tuple[float, float, float] = (1.25, 0.85, 0.55)
    rest_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tendon_moment_arms: tuple[float, float, float] = (1.00, 0.86, 0.68)
    min_angles: tuple[float, float, float] = field(
        default_factory=lambda: tuple(np.deg2rad([-95.0, -105.0, -90.0]))
    )
    max_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        link_lengths = _as_array3(self.link_lengths, "link_lengths")
        spring_stiffness = _as_array3(self.spring_stiffness, "spring_stiffness")
        rest_angles = _as_array3(self.rest_angles, "rest_angles")
        tendon_moment_arms = _as_array3(self.tendon_moment_arms, "tendon_moment_arms")
        min_angles = _as_array3(self.min_angles, "min_angles")
        max_angles = _as_array3(self.max_angles, "max_angles")

        if np.any(link_lengths <= 0.0):
            raise ValueError("link_lengths must be positive")
        if np.any(spring_stiffness <= 0.0):
            raise ValueError("spring_stiffness must be positive")
        if np.any(tendon_moment_arms < 0.0):
            raise ValueError("tendon_moment_arms must be nonnegative")
        if np.any(min_angles > max_angles):
            raise ValueError("min_angles must be <= max_angles elementwise")
        if np.any(rest_angles < min_angles) or np.any(rest_angles > max_angles):
            raise ValueError("rest_angles must lie within joint limits")

        object.__setattr__(self, "link_lengths", tuple(float(v) for v in link_lengths))
        object.__setattr__(self, "spring_stiffness", tuple(float(v) for v in spring_stiffness))
        object.__setattr__(self, "rest_angles", tuple(float(v) for v in rest_angles))
        object.__setattr__(self, "tendon_moment_arms", tuple(float(v) for v in tendon_moment_arms))
        object.__setattr__(self, "min_angles", tuple(float(v) for v in min_angles))
        object.__setattr__(self, "max_angles", tuple(float(v) for v in max_angles))

    @classmethod
    def from_fixed_config(cls, fixed_config, **kwargs) -> "TendonFingerConfig":
        """Create tendon parameters from a deterministic finger fixed config.

        ``fixed_config`` is intentionally duck-typed so this module stays
        independent from :mod:`finger.finger_config` at import time. It should
        expose ``OA``, ``AB``, and ``BD`` lengths.
        """
        return cls(
            link_lengths=(
                float(fixed_config.OA),
                float(fixed_config.AB),
                float(fixed_config.BD),
            ),
            **kwargs,
        )

    @classmethod
    def from_kinematic_model(cls, model, **kwargs) -> "TendonFingerConfig":
        """Create tendon parameters from a ``FingerKinematicModel`` geometry."""
        return cls.from_fixed_config(model.geometry.fixed, **kwargs)


@dataclass(frozen=True)
class TendonFingerState:
    """Solved quasi-static state for one tendon command."""

    command: float
    q: Array
    locked: tuple[bool, bool, bool]
    joint_positions: Array
    spring_torque: Array
    tendon_torque: Array
    net_joint_torque: Array
    contact_reaction_torque: Array
    joint_limit_reaction_torque: Array
    equilibrium_residual_torque: Array
    spring_energy: float
    tendon_potential: float

    @property
    def absolute_angles(self) -> Array:
        return np.cumsum(self.q)

    @property
    def tip(self) -> Array:
        return self.joint_positions[-1]

    @property
    def total_potential(self) -> float:
        return self.spring_energy + self.tendon_potential

    @property
    def active_contact_joints(self) -> tuple[str, ...]:
        names = ("mcp", "pip", "dip")
        return tuple(name for name, is_locked in zip(names, self.locked) if is_locked)

    @property
    def max_abs_equilibrium_residual_torque(self) -> float:
        return float(np.max(np.abs(self.equilibrium_residual_torque)))

    @property
    def total_reaction_torque(self) -> Array:
        return self.contact_reaction_torque + self.joint_limit_reaction_torque

    def point_dict(self) -> dict[str, Array]:
        return {
            "base": self.joint_positions[0],
            "mcp": self.joint_positions[0],
            "pip": self.joint_positions[1],
            "dip": self.joint_positions[2],
            "tip": self.joint_positions[3],
        }


@dataclass(frozen=True)
class TendonFingerTrajectory:
    """Command sweep results for the tendon-driven model."""

    states: tuple[TendonFingerState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))

    @property
    def q(self) -> Array:
        return np.vstack([state.q for state in self.states])

    @property
    def tips(self) -> Array:
        return np.vstack([state.tip for state in self.states])

    @property
    def joint_positions(self) -> Array:
        return np.stack([state.joint_positions for state in self.states], axis=0)

    @property
    def commands(self) -> Array:
        return np.array([state.command for state in self.states], dtype=float)


class TendonFinger:
    """Quasi-static underactuated 3R finger driven by one tendon command.

    In free space, the equilibrium is analytical because the joints are only
    coupled through the shared tendon command:

        E(q, u) = 0.5 * sum(k_i * (q_i - q0_i)^2)
                  + u * sum(r_i * q_i)

    Since flexion is negative in this repository's figures, positive ``u``
    lowers the energy by making ``q`` more negative. When a phalanx contacts an
    object, use ``contact_locks`` and ``locked_angles`` to hold the
    corresponding joint at the angle where contact occurred. MCP, PIP, and DIP
    can lock independently, or all together. The unlocked joints keep
    responding to the tendon command, while locked joints report the contact
    reaction torque needed to balance tendon and spring torques.
    """

    joint_names = ("mcp", "pip", "dip")

    def __init__(self, config: TendonFingerConfig | None = None) -> None:
        self.config = TendonFingerConfig() if config is None else config
        self.link_lengths = _as_array3(self.config.link_lengths, "link_lengths")
        self.spring_stiffness = _as_array3(self.config.spring_stiffness, "spring_stiffness")
        self.rest_angles = _as_array3(self.config.rest_angles, "rest_angles")
        self.tendon_moment_arms = _as_array3(self.config.tendon_moment_arms, "tendon_moment_arms")
        self.min_angles = _as_array3(self.config.min_angles, "min_angles")
        self.max_angles = _as_array3(self.config.max_angles, "max_angles")

    @classmethod
    def from_kinematic_model(cls, model, **config_kwargs) -> "TendonFinger":
        """Build a tendon finger with the same phalanx lengths as a linkage model."""
        return cls(TendonFingerConfig.from_kinematic_model(model, **config_kwargs))

    def free_equilibrium(self, command: float) -> Array:
        """Return the unloaded torsional-spring equilibrium for one command."""
        command = float(command)
        q = self.rest_angles - command * self.tendon_moment_arms / self.spring_stiffness
        return np.clip(q, self.min_angles, self.max_angles)

    def forward_kinematics(self, q: Sequence[float]) -> Array:
        """Return base, PIP, DIP, and tip positions for relative joint angles."""
        q_arr = _as_array3(q, "q")
        absolute_angles = np.cumsum(q_arr)
        link_vectors = np.column_stack(
            (
                self.link_lengths * np.cos(absolute_angles),
                self.link_lengths * np.sin(absolute_angles),
            )
        )
        return np.vstack((np.zeros(2), np.cumsum(link_vectors, axis=0)))

    def tip_position(self, q: Sequence[float]) -> Array:
        """Return the fingertip position for relative joint angles."""
        return self.forward_kinematics(q)[-1]

    def tendon_torque(self, command: float) -> Array:
        """Torque applied by the tendon; negative means flexion in this convention."""
        return -float(command) * self.tendon_moment_arms

    def spring_torque(self, q: Sequence[float]) -> Array:
        """Restoring torque from torsional springs."""
        q_arr = _as_array3(q, "q")
        return -self.spring_stiffness * (q_arr - self.rest_angles)

    def net_joint_torque(self, q: Sequence[float], command: float) -> Array:
        """Return spring plus tendon torque before contact/limit reactions."""
        return self.spring_torque(q) + self.tendon_torque(command)

    def contact_reaction_torque(
        self,
        q: Sequence[float],
        command: float,
        contact_locks: Sequence[bool],
    ) -> Array:
        """Torque supplied by contact constraints at locked joints."""
        q_arr = _as_array3(q, "q")
        locks = np.asarray(contact_locks, dtype=bool).reshape(-1)
        if locks.size != 3:
            raise ValueError(f"contact_locks must contain 3 flags, got {locks.size}")
        net = self.net_joint_torque(q_arr, command)
        reaction = np.zeros(3, dtype=float)
        reaction[locks] = -net[locks]
        return reaction

    def joint_limit_reaction_torque(
        self,
        q: Sequence[float],
        command: float,
        contact_locks: Sequence[bool] | None = None,
        tol: float = 1e-10,
    ) -> Array:
        """Torque supplied by joint limits for unlocked saturated joints."""
        q_arr = _as_array3(q, "q")
        if contact_locks is None:
            locks = np.zeros(3, dtype=bool)
        else:
            locks = np.asarray(contact_locks, dtype=bool).reshape(-1)
            if locks.size != 3:
                raise ValueError(f"contact_locks must contain 3 flags, got {locks.size}")

        net = self.net_joint_torque(q_arr, command)
        lower_active = (~locks) & (q_arr <= self.min_angles + tol) & (net < 0.0)
        upper_active = (~locks) & (q_arr >= self.max_angles - tol) & (net > 0.0)
        reaction = np.zeros(3, dtype=float)
        reaction[lower_active | upper_active] = -net[lower_active | upper_active]
        return reaction

    def spring_energy(self, q: Sequence[float]) -> float:
        """Torsional spring energy."""
        q_arr = _as_array3(q, "q")
        dq = q_arr - self.rest_angles
        return float(0.5 * np.sum(self.spring_stiffness * dq * dq))

    def tendon_potential(self, q: Sequence[float], command: float) -> float:
        """Potential term whose gradient equals the tendon generalized force sign."""
        q_arr = _as_array3(q, "q")
        return float(command * np.dot(self.tendon_moment_arms, q_arr))

    def solve(
        self,
        command: float,
        contact_locks: Sequence[bool] | None = None,
        locked_angles: Sequence[float] | None = None,
    ) -> TendonFingerState:
        """Solve the quasi-static pose for one command.

        Parameters
        ----------
        command:
            Nonnegative tendon command. Larger values flex the finger.
        contact_locks:
            Boolean flags for MCP/PIP/DIP. Any subset can lock. A locked joint
            stays fixed instead of following the free-space tendon/spring
            equilibrium.
        locked_angles:
            Joint angles used by locked joints. If omitted, locked joints hold
            their rest angles.
        """
        command = float(command)
        if command < 0.0:
            raise ValueError(f"command must be nonnegative, got {command}")

        q = self.free_equilibrium(command)
        if contact_locks is None:
            locks = np.zeros(3, dtype=bool)
        else:
            locks = np.asarray(contact_locks, dtype=bool).reshape(-1)
            if locks.size != 3:
                raise ValueError(f"contact_locks must contain 3 flags, got {locks.size}")

        if np.any(locks):
            if locked_angles is None:
                locked = self.rest_angles.copy()
            else:
                locked = _as_array3(locked_angles, "locked_angles")
            q[locks] = locked[locks]
            q = np.clip(q, self.min_angles, self.max_angles)

        joints = self.forward_kinematics(q)
        spring_torque = self.spring_torque(q)
        tendon_torque = self.tendon_torque(command)
        net_joint_torque = spring_torque + tendon_torque
        contact_reaction_torque = self.contact_reaction_torque(q, command, locks)
        joint_limit_reaction_torque = self.joint_limit_reaction_torque(q, command, locks)
        equilibrium_residual_torque = (
            net_joint_torque
            + contact_reaction_torque
            + joint_limit_reaction_torque
        )
        return TendonFingerState(
            command=command,
            q=q,
            locked=tuple(bool(v) for v in locks),
            joint_positions=joints,
            spring_torque=spring_torque,
            tendon_torque=tendon_torque,
            net_joint_torque=net_joint_torque,
            contact_reaction_torque=contact_reaction_torque,
            joint_limit_reaction_torque=joint_limit_reaction_torque,
            equilibrium_residual_torque=equilibrium_residual_torque,
            spring_energy=self.spring_energy(q),
            tendon_potential=self.tendon_potential(q, command),
        )

    def solve_many(
        self,
        commands: Sequence[float],
        contact_locks: Sequence[bool] | None = None,
        locked_angles: Sequence[float] | None = None,
    ) -> TendonFingerTrajectory:
        """Solve an independent list of commands with the same lock pattern."""
        states = tuple(self.solve(command, contact_locks, locked_angles) for command in commands)
        return TendonFingerTrajectory(states)

    def simulate_contact_sequence(
        self,
        commands: Sequence[float],
        contact_commands: Mapping[int | str, float] | None = None,
    ) -> TendonFingerTrajectory:
        """Simulate contact events that lock joints at their first-contact angle.

        ``contact_commands`` maps joint ids to the command value where that
        joint first contacts the object. Accepted keys are 0/1/2 or
        ``"mcp"``, ``"pip"``, ``"dip"``. All three joints may appear in the
        mapping, and multiple joints may share the same threshold. Once active,
        the rule is:

            contact exists -> that q stays at the contact angle

        This intentionally creates path divergence compared with the unloaded
        trajectory, while preserving the one-command tendon actuation.
        """
        commands_arr = np.asarray(commands, dtype=float).reshape(-1)
        if np.any(commands_arr < 0.0):
            raise ValueError("commands must be nonnegative")

        thresholds = self._contact_thresholds(contact_commands)
        locks = np.zeros(3, dtype=bool)
        locked_angles = self.rest_angles.copy()
        states: list[TendonFingerState] = []

        for command in commands_arr:
            for joint_idx, threshold in thresholds.items():
                if not locks[joint_idx] and command >= threshold:
                    locked_angles[joint_idx] = self.free_equilibrium(threshold)[joint_idx]
                    locks[joint_idx] = True
            states.append(self.solve(float(command), locks, locked_angles))

        return TendonFingerTrajectory(tuple(states))

    def _contact_thresholds(self, contact_commands: Mapping[int | str, float] | None) -> dict[int, float]:
        if contact_commands is None:
            return {}

        thresholds: dict[int, float] = {}
        for key, value in contact_commands.items():
            if isinstance(key, str):
                try:
                    idx = self.joint_names.index(key.lower())
                except ValueError as exc:
                    raise KeyError(f"Unknown joint name {key!r}; use mcp, pip, or dip") from exc
            else:
                idx = int(key)
            if idx < 0 or idx >= 3:
                raise IndexError(f"joint index must be 0, 1, or 2, got {idx}")
            threshold = float(value)
            if threshold < 0.0:
                raise ValueError(f"contact command must be nonnegative, got {threshold}")
            thresholds[idx] = threshold
        return thresholds
