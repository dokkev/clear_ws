"""
1-DoF / 3-joint planar finger linkage demo.

Kinematic idea:
  theta1 is the only input.
  The middle and distal phalanx poses are solved by two circle-circle constraints:

    1) B' is fixed on the middle body and connected to the ground pivot O'
       by a rigid link.
    2) C' is fixed on the distal body and connected to A' on the proximal body
       by a rigid link.

This script generates:
  - finger_1dof_3joint_animation.gif
  - finger_1dof_3joint_trajectory.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np

try:
    from finger.finger_config import FingerDesignVariables, FingerExtendedGeometry, FingerFixedConfig
    from finger.kinematic_model import FingerKinematicModel, FingerState
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from finger.finger_config import FingerDesignVariables, FingerExtendedGeometry, FingerFixedConfig
    from finger.kinematic_model import FingerKinematicModel, FingerState


OUT_DIR = Path(__file__).resolve().parents[1] / "output"

COLORS = {
    "ground": "#555555",
    "proximal": "#e41a1c",
    "middle": "#377eb8",
    "distal": "#4daf4a",
    "transmission": "#222222",
    "trajectory": "#ff7f00",
}


def configure_figure_style() -> str:
    """Set a compact serif style suitable for proposal figures."""
    serif_candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_font = next((font for font in serif_candidates if font in available_fonts), "DejaVu Serif")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif_font, *[font for font in serif_candidates if font != serif_font]],
        "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.75,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "#cfcfcf",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "savefig.dpi": 300,
    })
    return serif_font


def apply_ieee_axes_style(ax) -> None:
    """Apply light grid and thin axes styling."""
    ax.set_axisbelow(True)
    ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#111111")
    ax.tick_params(which="both", top=False, right=False, width=0.7)


def R(theta: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def angle_of(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def wrap_to_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def deg(theta: float) -> float:
    return math.degrees(wrap_to_pi(theta))


def circle_intersections(
    c0: np.ndarray,
    r0: float,
    c1: np.ndarray,
    r1: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Return the two intersections of circles:
      ||x - c0|| = r0
      ||x - c1|| = r1

    Shape: (2, 2). Raises ValueError if no real intersection exists.
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)

    d_vec = c1 - c0
    d = np.linalg.norm(d_vec)

    if d < eps:
        raise ValueError("Circle centers are coincident; branch is under-defined.")
    if d > r0 + r1 + eps:
        raise ValueError(f"No intersection: circles too far apart. d={d:.4f}")
    if d < abs(r0 - r1) - eps:
        raise ValueError(f"No intersection: one circle lies inside the other. d={d:.4f}")

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = r0 * r0 - a * a
    h = math.sqrt(max(0.0, h_sq))

    e = d_vec / d
    p = c0 + a * e
    n = np.array([-e[1], e[0]], dtype=float)

    return np.vstack((p + h * n, p - h * n))


def choose_continuous(
    candidates: np.ndarray,
    previous_point: np.ndarray | None,
    prefer_lower_y: bool = True,
) -> np.ndarray:
    """
    Pick a branch from two circle-intersection candidates.

    For the first frame, use a deterministic branch choice.
    After that, choose the candidate closest to the previous point.
    """
    if previous_point is None:
        idx = int(np.argmin(candidates[:, 1])) if prefer_lower_y else 0
        return candidates[idx]

    distances = np.linalg.norm(candidates - previous_point[None, :], axis=1)
    return candidates[int(np.argmin(distances))]


@dataclass
class FingerGeometry:
    """
    All body-attached points are defined in each phalanx local frame.

    Proximal body frame:
      origin at O
      A_local, Ap_local are fixed points on the proximal body.

    Middle body frame:
      origin at A
      B_local, Bp_local are fixed points on the middle body.

    Distal body frame:
      origin at B
      Cp_local, tip_local are fixed points on the distal body.
    """

    O: np.ndarray
    Op: np.ndarray
    A_local: np.ndarray
    Ap_local: np.ndarray
    B_local: np.ndarray
    Bp_local: np.ndarray
    Cp_local: np.ndarray
    tip_local: np.ndarray

    L_OpBp: float = 0.0
    L_ApCp: float = 0.0
    r_ABp: float = 0.0
    r_BCp: float = 0.0
    alpha_Bp: float = 0.0
    alpha_Cp: float = 0.0

    def __post_init__(self) -> None:
        self.O = np.asarray(self.O, dtype=float)
        self.Op = np.asarray(self.Op, dtype=float)
        self.A_local = np.asarray(self.A_local, dtype=float)
        self.Ap_local = np.asarray(self.Ap_local, dtype=float)
        self.B_local = np.asarray(self.B_local, dtype=float)
        self.Bp_local = np.asarray(self.Bp_local, dtype=float)
        self.Cp_local = np.asarray(self.Cp_local, dtype=float)
        self.tip_local = np.asarray(self.tip_local, dtype=float)

        A0 = self.O + self.A_local
        Ap0 = self.O + self.Ap_local
        B0 = A0 + self.B_local
        Bp0 = A0 + self.Bp_local
        Cp0 = B0 + self.Cp_local

        self.L_OpBp = float(np.linalg.norm(Bp0 - self.Op))
        self.L_ApCp = float(np.linalg.norm(Cp0 - Ap0))
        self.r_ABp = float(np.linalg.norm(self.Bp_local))
        self.r_BCp = float(np.linalg.norm(self.Cp_local))
        self.alpha_Bp = angle_of(self.Bp_local)
        self.alpha_Cp = angle_of(self.Cp_local)


@dataclass
class FingerPose:
    theta1: float
    theta2: float
    theta3: float

    O: np.ndarray
    Op: np.ndarray
    A: np.ndarray
    Ap: np.ndarray
    Bp: np.ndarray
    B: np.ndarray
    Cp: np.ndarray
    tip: np.ndarray


def build_demo_finger_model() -> tuple[FingerExtendedGeometry, FingerKinematicModel]:
    """Build the finger package model that matches the original demo geometry."""
    fixed = FingerFixedConfig(OA=2.00, AB=1.75, BD=1.55)
    design = FingerDesignVariables(
        OO_prime=(0.70, 1.20),
        AA_prime=(0.60, 0.55),
        AB_prime=(0.05, -0.65),
        BC_prime=(0.25, -0.45),
    )
    geometry = FingerExtendedGeometry(fixed=fixed, design=design)
    return geometry, FingerKinematicModel(geometry)


def pose_from_finger_state(state: FingerState) -> FingerPose:
    """Adapt the package FingerState to the plotting helpers in this file."""
    return FingerPose(
        theta1=state.q1,
        theta2=state.q2,
        theta3=state.q3,
        O=state.O,
        Op=state.O_prime,
        A=state.A,
        Ap=state.A_prime,
        Bp=state.B_prime,
        B=state.B,
        Cp=state.C_prime,
        tip=state.D,
    )


def relative_joint_angles(pose: FingerPose) -> tuple[float, float, float]:
    """
    Return relative joint angles:
      q1: proximal relative to ground
      q2: middle relative to proximal
      q3: distal relative to middle
    """
    q1 = wrap_to_pi(pose.theta1)
    q2 = wrap_to_pi(pose.theta2 - pose.theta1)
    q3 = wrap_to_pi(pose.theta3 - pose.theta2)
    return q1, q2, q3


def body_triangles(pose: FingerPose) -> dict[str, np.ndarray]:
    """Return rigid-body triangle vertices keyed by body color name."""
    return {
        "proximal": np.vstack((pose.O, pose.A, pose.Ap)),
        "middle": np.vstack((pose.A, pose.B, pose.Bp)),
        "distal": np.vstack((pose.B, pose.tip, pose.Cp)),
    }


def draw_body_fills(ax, pose: FingerPose, alpha: float, zorder: float) -> None:
    """Fill O-A-A', A-B-B', and B-C-C' behind their outlines."""
    for name, points in body_triangles(pose).items():
        ax.fill(
            points[:, 0],
            points[:, 1],
            facecolor=COLORS[name],
            edgecolor="none",
            alpha=alpha,
            label="_nolegend_",
            zorder=zorder,
        )


def solve_pose(
    theta1: float,
    geom: FingerGeometry,
    previous_pose: FingerPose | None = None,
) -> FingerPose:
    """Given theta1, solve theta2 and theta3 from linkage constraints."""
    R1 = R(theta1)

    O = geom.O
    Op = geom.Op

    A = O + R1 @ geom.A_local
    Ap = O + R1 @ geom.Ap_local

    prev_Bp = None if previous_pose is None else previous_pose.Bp
    prev_Cp = None if previous_pose is None else previous_pose.Cp

    Bp_candidates = circle_intersections(A, geom.r_ABp, Op, geom.L_OpBp)
    Bp = choose_continuous(Bp_candidates, prev_Bp, prefer_lower_y=True)

    theta2 = angle_of(Bp - A) - geom.alpha_Bp
    R2 = R(theta2)
    B = A + R2 @ geom.B_local

    Cp_candidates = circle_intersections(B, geom.r_BCp, Ap, geom.L_ApCp)
    Cp = choose_continuous(Cp_candidates, prev_Cp, prefer_lower_y=True)

    theta3 = angle_of(Cp - B) - geom.alpha_Cp
    R3 = R(theta3)
    tip = B + R3 @ geom.tip_local

    return FingerPose(
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        O=O,
        Op=Op,
        A=A,
        Ap=Ap,
        Bp=Bp,
        B=B,
        Cp=Cp,
        tip=tip,
    )


def simulate(theta_values: np.ndarray, model: FingerKinematicModel) -> list[FingerPose]:
    """
    Sweep theta1 and keep only feasible poses.
    Branch continuity is maintained using the previous valid pose.
    """
    trajectory = model.solve_trajectory(theta_values)
    poses = [pose_from_finger_state(state) for state in trajectory.states]

    if len(poses) < 2:
        raise RuntimeError("Too few feasible poses. Adjust geometry or theta sweep.")

    return poses


def draw_pose_context(
    ax,
    pose: FingerPose,
    linewidth: float = 1.6,
    alpha: float = 0.45,
    zorder: int = 2,
    show_labels: bool = False,
) -> None:
    """Draw a pose for trajectory context."""
    def label(text: str) -> str:
        return text if show_labels else "_nolegend_"

    draw_body_fills(ax, pose, alpha=min(0.30, alpha * 0.36), zorder=zorder)

    ax.plot(
        [pose.O[0], pose.Op[0]],
        [pose.O[1], pose.Op[1]],
        "o",
        color=COLORS["ground"],
        alpha=alpha,
        label=label("ground pivots"),
        zorder=zorder,
    )

    prox = np.vstack((pose.O, pose.A, pose.Ap, pose.O))
    mid = np.vstack((pose.A, pose.B, pose.Bp, pose.A))
    dist = np.vstack((pose.B, pose.tip, pose.Cp, pose.B))

    ax.plot(
        prox[:, 0],
        prox[:, 1],
        "o-",
        color=COLORS["proximal"],
        linewidth=linewidth,
        alpha=alpha,
        label=label("proximal body"),
        zorder=zorder,
    )
    ax.plot(
        mid[:, 0],
        mid[:, 1],
        "o-",
        color=COLORS["middle"],
        linewidth=linewidth,
        alpha=alpha,
        label=label("middle body"),
        zorder=zorder,
    )
    ax.plot(
        dist[:, 0],
        dist[:, 1],
        "o-",
        color=COLORS["distal"],
        linewidth=linewidth,
        alpha=alpha,
        label=label("distal body"),
        zorder=zorder,
    )
    ax.plot(
        [pose.Op[0], pose.Bp[0]],
        [pose.Op[1], pose.Bp[1]],
        "-",
        color=COLORS["transmission"],
        linewidth=linewidth,
        alpha=alpha,
        label=label("coupler links"),
        zorder=zorder,
    )
    ax.plot(
        [pose.Ap[0], pose.Cp[0]],
        [pose.Ap[1], pose.Cp[1]],
        "-",
        color=COLORS["transmission"],
        linewidth=linewidth,
        alpha=alpha,
        label="_nolegend_",
        zorder=zorder,
    )


def make_animation(
    poses: list[FingerPose],
    out_path: Path,
    fps: int = 24,
) -> None:
    configure_figure_style()
    tips = np.vstack([p.tip for p in poses])

    all_points = []
    for p in poses:
        all_points.extend([p.O, p.Op, p.A, p.Ap, p.Bp, p.B, p.Cp, p.tip])
    all_points = np.vstack(all_points)

    margin = 0.35
    xmin, ymin = np.min(all_points, axis=0) - margin
    xmax, ymax = np.max(all_points, axis=0) + margin

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    apply_ieee_axes_style(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("1-DoF / 3-joint finger linkage")

    fill_alpha = 0.22
    fill_patches = {
        name: Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor=COLORS[name],
            edgecolor="none",
            alpha=fill_alpha,
            zorder=1,
        )
        for name in ("proximal", "middle", "distal")
    }
    for patch in fill_patches.values():
        ax.add_patch(patch)

    ground_line, = ax.plot([], [], "o", color=COLORS["ground"], label="ground pivots")
    prox_line, = ax.plot([], [], "o-", color=COLORS["proximal"], linewidth=2.5, label="proximal body", zorder=3)
    mid_line, = ax.plot([], [], "o-", color=COLORS["middle"], linewidth=2.5, label="middle body", zorder=3)
    dist_line, = ax.plot([], [], "o-", color=COLORS["distal"], linewidth=2.5, label="distal body", zorder=3)
    link1_line, = ax.plot([], [], "-", color=COLORS["transmission"], linewidth=1.5, label="O'-B' coupler", zorder=4)
    link2_line, = ax.plot([], [], "-", color=COLORS["transmission"], linewidth=1.5, label="A'-C' coupler", zorder=4)
    traj_line, = ax.plot([], [], "-", color=COLORS["trajectory"], linewidth=1.5, label="fingertip trajectory", zorder=5)
    tip_marker, = ax.plot([], [], "o", color=COLORS["trajectory"], markersize=7, label="tip", zorder=6)

    theta_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=8)

    def set_line(line, pts):
        pts = np.asarray(pts, dtype=float)
        line.set_data(pts[:, 0], pts[:, 1])

    def update(i: int):
        p = poses[i]

        for name, points in body_triangles(p).items():
            fill_patches[name].set_xy(points)

        set_line(ground_line, [p.O, p.Op])
        set_line(prox_line, [p.O, p.A, p.Ap, p.O])
        set_line(mid_line, [p.A, p.B, p.Bp, p.A])
        set_line(dist_line, [p.B, p.tip, p.Cp, p.B])
        set_line(link1_line, [p.Op, p.Bp])
        set_line(link2_line, [p.Ap, p.Cp])
        set_line(traj_line, tips[: i + 1])
        set_line(tip_marker, [p.tip])

        q1, q2, q3 = relative_joint_angles(p)
        theta_text.set_text(
            f"q1 = {math.degrees(q1): .1f} deg\n"
            f"q2 = {math.degrees(q2): .1f} deg\n"
            f"q3 = {math.degrees(q3): .1f} deg"
        )

        return (
            *fill_patches.values(),
            ground_line,
            prox_line,
            mid_line,
            dist_line,
            link1_line,
            link2_line,
            traj_line,
            tip_marker,
            theta_text,
        )

    ani = FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps, blit=True)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_trajectory_plot(
    poses: list[FingerPose],
    out_path: Path,
) -> None:
    configure_figure_style()
    tips = np.vstack([p.tip for p in poses])
    theta_deg = np.array([deg(p.theta1) for p in poses])

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    apply_ieee_axes_style(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Deterministic fingertip trajectory from input sweep")

    draw_pose_context(ax, poses[0], linewidth=1.5, alpha=0.22, zorder=1)
    draw_pose_context(ax, poses[-1], linewidth=2.2, alpha=0.92, zorder=2, show_labels=True)

    ax.plot(
        tips[:, 0],
        tips[:, 1],
        "-",
        color=COLORS["trajectory"],
        linewidth=2.4,
        label="deterministic tip trajectory",
        zorder=4,
    )
    ax.scatter(
        tips[0, 0],
        tips[0, 1],
        marker="o",
        s=60,
        facecolor="white",
        edgecolor=COLORS["trajectory"],
        linewidth=1.6,
        label=f"start q1={theta_deg[0]:.1f} deg",
        zorder=6,
    )
    ax.scatter(
        tips[-1, 0],
        tips[-1, 1],
        marker="s",
        s=60,
        color=COLORS["trajectory"],
        edgecolor="black",
        linewidth=0.75,
        label=f"end q1={theta_deg[-1]:.1f} deg",
        zorder=6,
    )

    ax.text(tips[0, 0], tips[0, 1], "  start", va="center")
    ax.text(tips[-1, 0], tips[-1, 1], "  end", va="center")
    ax.legend(fontsize=9, loc="best")

    all_points = []
    for p in poses:
        all_points.extend([p.O, p.Op, p.A, p.Ap, p.Bp, p.B, p.Cp, p.tip])
    all_points = np.vstack(all_points)
    margin = 0.45
    xmin, ymin = np.min(all_points, axis=0) - margin
    xmax, ymax = np.max(all_points, axis=0) + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    geometry, finger_model = build_demo_finger_model()

    theta_values = np.deg2rad(np.linspace(0.0, -40.0, 120))
    poses = simulate(theta_values, finger_model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = OUT_DIR / "finger_1dof_3joint_animation.gif"
    png_path = OUT_DIR / "finger_1dof_3joint_trajectory.png"

    make_animation(poses, gif_path, fps=24)
    make_trajectory_plot(poses, png_path)

    print("Solved poses:", len(poses))
    print(f"O'-B' link length: {geometry.O_prime_B_prime_length:.4f}")
    print(f"A'-C' link length: {geometry.A_prime_C_prime_length:.4f}")
    q_start = relative_joint_angles(poses[0])
    q_end = relative_joint_angles(poses[-1])
    print(f"q1 range: {math.degrees(q_start[0]):.2f} to {math.degrees(q_end[0]):.2f} deg")
    print(f"q2 range: {math.degrees(q_start[1]):.2f} to {math.degrees(q_end[1]):.2f} deg")
    print(f"q3 range: {math.degrees(q_start[2]):.2f} to {math.degrees(q_end[2]):.2f} deg")
    print("Saved:", gif_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
