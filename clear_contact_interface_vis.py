"""
Contact-interface visualization for a deterministic linkage finger.

The gross finger pose is prescribed by q. Contact does not change q; it only
deforms proximal/middle compliant sleeve nodes. The distal phalanx is drawn as
a rigid cap and is excluded from the compliant contact network.

Run:
    python clear_contact_interface_vis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from scipy.optimize import minimize


REQUESTED_OUT_DIR = Path("/mnt/data/clear_contact_interface_outputs")
FALLBACK_OUT_DIR = Path(__file__).resolve().parent / "output" / "clear_contact_interface_outputs"


def prepare_output_dir() -> Path:
    try:
        REQUESTED_OUT_DIR.mkdir(parents=True, exist_ok=True)
        return REQUESTED_OUT_DIR
    except PermissionError:
        FALLBACK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Warning: cannot write {REQUESTED_OUT_DIR}; using {FALLBACK_OUT_DIR}")
        return FALLBACK_OUT_DIR


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n > 1e-12:
        return v / n
    if fallback is None:
        return np.array([1.0, 0.0], dtype=float)
    return np.asarray(fallback, dtype=float)


def as_points(points: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(points, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[None, :]
    return arr, was_1d


def set_research_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.linewidth": 0.75,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
    })


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    rest: float
    k: float
    kind: str


@dataclass
class CircleObject:
    center: np.ndarray
    radius: float
    label: str = "circle"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts, was_1d = as_points(points)
        phi = np.linalg.norm(pts - self.center[None, :], axis=1) - self.radius
        return phi[0] if was_1d else phi

    def normal(self, points: np.ndarray) -> np.ndarray:
        pts, was_1d = as_points(points)
        vec = pts - self.center[None, :]
        nrm = np.linalg.norm(vec, axis=1)
        normals = np.zeros_like(vec)
        good = nrm > 1e-12
        normals[good] = vec[good] / nrm[good, None]
        normals[~good] = np.array([1.0, 0.0])
        return normals[0] if was_1d else normals

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        pts, was_1d = as_points(points)
        cps = self.center[None, :] + self.radius * self.normal(pts)
        return cps[0] if was_1d else cps

    def with_center(self, center: np.ndarray) -> "CircleObject":
        return CircleObject(np.asarray(center, dtype=float), self.radius, self.label)

    def draw(self, ax, alpha: float = 0.62, zorder: float = 1.2) -> None:
        ax.add_patch(Circle(
            self.center,
            self.radius,
            facecolor="#d7d7d7",
            edgecolor="#343434",
            linewidth=0.9,
            alpha=alpha,
            zorder=zorder,
        ))


@dataclass
class BoxObject:
    center: np.ndarray
    width: float
    height: float
    rotation: float = 0.0
    label: str = "box"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    @property
    def R(self) -> np.ndarray:
        return rotation(self.rotation)

    @property
    def half_extents(self) -> np.ndarray:
        return np.array([0.5 * self.width, 0.5 * self.height], dtype=float)

    def _local(self, points: np.ndarray) -> tuple[np.ndarray, bool]:
        pts, was_1d = as_points(points)
        return (pts - self.center[None, :]) @ self.R, was_1d

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        q, was_1d = self._local(points)
        h = self.half_extents
        d = np.abs(q) - h[None, :]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0)
        phi = outside + inside
        return phi[0] if was_1d else phi

    def normal(self, points: np.ndarray) -> np.ndarray:
        q, was_1d = self._local(points)
        h = self.half_extents
        d = np.abs(q) - h[None, :]
        outside = np.any(d > 0.0, axis=1)
        normals_local = np.zeros_like(q)

        if np.any(outside):
            clipped = np.clip(q[outside], -h[None, :], h[None, :])
            vec = q[outside] - clipped
            norms = np.linalg.norm(vec, axis=1)
            good = norms > 1e-12
            normals_local[outside][good] = vec[good] / norms[good, None]

        inside = ~outside
        if np.any(inside):
            dist_to_face = h[None, :] - np.abs(q[inside])
            axes = np.argmin(dist_to_face, axis=1)
            local_inside = np.zeros((np.sum(inside), 2), dtype=float)
            for row, axis in enumerate(axes):
                sign = 1.0 if q[inside][row, axis] >= 0.0 else -1.0
                local_inside[row, axis] = sign
            normals_local[inside] = local_inside

        # Fix possible chained-indexing gaps for outside points.
        if np.any(outside):
            outside_idx = np.where(outside)[0]
            clipped = np.clip(q[outside], -h[None, :], h[None, :])
            vec = q[outside] - clipped
            norms = np.linalg.norm(vec, axis=1)
            for row, idx in enumerate(outside_idx):
                if norms[row] > 1e-12:
                    normals_local[idx] = vec[row] / norms[row]
                else:
                    normals_local[idx] = np.array([1.0, 0.0])

        normals_world = normals_local @ self.R.T
        return normals_world[0] if was_1d else normals_world

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        q, was_1d = self._local(points)
        h = self.half_extents
        cp = np.clip(q, -h[None, :], h[None, :])
        inside = np.all(np.abs(q) <= h[None, :], axis=1)
        if np.any(inside):
            inside_idx = np.where(inside)[0]
            dist_to_face = h[None, :] - np.abs(q[inside])
            axes = np.argmin(dist_to_face, axis=1)
            for row, idx in enumerate(inside_idx):
                axis = axes[row]
                sign = 1.0 if q[idx, axis] >= 0.0 else -1.0
                cp[idx] = q[idx]
                cp[idx, axis] = sign * h[axis]
        cps = self.center[None, :] + cp @ self.R.T
        return cps[0] if was_1d else cps

    def with_center(self, center: np.ndarray) -> "BoxObject":
        return BoxObject(np.asarray(center, dtype=float), self.width, self.height, self.rotation, self.label)

    def vertices(self) -> np.ndarray:
        h = self.half_extents
        local = np.array([
            [-h[0], -h[1]],
            [h[0], -h[1]],
            [h[0], h[1]],
            [-h[0], h[1]],
        ])
        return self.center[None, :] + local @ self.R.T

    def draw(self, ax, alpha: float = 0.62, zorder: float = 1.2) -> None:
        ax.add_patch(Polygon(
            self.vertices(),
            closed=True,
            facecolor="#d7d7d7",
            edgecolor="#343434",
            linewidth=0.9,
            alpha=alpha,
            zorder=zorder,
        ))


class HingeCompatibleSleeveV4:
    """Planar sleeve model over proximal/middle links with a rigid distal cap."""

    def __init__(self, mode: str = "routed", n_segments: int = 48) -> None:
        if mode not in {"naive", "routed"}:
            raise ValueError("mode must be 'naive' or 'routed'")
        self.mode = mode
        self.N = int(n_segments)
        self.link_lengths = np.array([34.0, 28.0, 20.0], dtype=float)
        self.pad_inner_offset = -1.0
        self.pad_outer_offset = -7.0
        self.half_width = 4.4
        self.distal_half_width = 4.8
        # The compliant contact pad stops before the DIP/distal cap. This keeps
        # distal phalanx adaptation rigid-only and avoids a distal-attached pad.
        self.middle_pad_fraction = 0.78
        self.sleeve_length = float(self.link_lengths[0] + self.middle_pad_fraction * self.link_lengths[1])
        self.hinge_positions = np.array([0.0, self.link_lengths[0]], dtype=float)
        self.s_values = np.linspace(0.0, self.sleeve_length, self.N + 1)
        self.top_ids = np.arange(0, self.N + 1)
        self.bottom_ids = np.arange(self.N + 1, 2 * (self.N + 1))
        self.local_nodes = self._make_local_nodes()
        self.edges = self._make_edges()
        self.k_anchor = 0.052 if mode == "routed" else 0.075
        self.k_contact_default = 2600.0

    def _make_local_nodes(self) -> np.ndarray:
        top = np.column_stack((self.s_values, np.full(self.N + 1, self.pad_inner_offset)))
        bottom = np.column_stack((self.s_values, np.full(self.N + 1, self.pad_outer_offset)))
        return np.vstack((top, bottom))

    def _edge_rest(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.local_nodes[i] - self.local_nodes[j]))

    def _interval_crosses_pip(self, station: int) -> bool:
        s0, s1 = self.s_values[station], self.s_values[station + 1]
        return s0 <= self.link_lengths[0] <= s1 or abs(0.5 * (s0 + s1) - self.link_lengths[0]) < 1.5

    def _make_edges(self) -> list[Edge]:
        edges: list[Edge] = []
        for k in range(self.N):
            pip_cross = self._interval_crosses_pip(k)
            top0, top1 = self.top_ids[k], self.top_ids[k + 1]
            bot0, bot1 = self.bottom_ids[k], self.bottom_ids[k + 1]

            rail_k = 1.0 if self.mode == "naive" or not pip_cross else 0.72
            diag_k = 0.36 if self.mode == "naive" else 0.20
            if pip_cross and self.mode == "routed":
                diag_k = 0.045
            kind_diag = "diag_hinge" if pip_cross else "diag"

            edges.append(Edge(top0, top1, self._edge_rest(top0, top1), rail_k, "top_rail"))
            edges.append(Edge(bot0, bot1, self._edge_rest(bot0, bot1), rail_k, "bottom_rail"))
            edges.append(Edge(top0, bot0, self._edge_rest(top0, bot0), 0.54, "cross"))
            edges.append(Edge(top0, bot1, self._edge_rest(top0, bot1), diag_k, kind_diag))
            edges.append(Edge(bot0, top1, self._edge_rest(bot0, top1), diag_k, kind_diag))
        edges.append(Edge(self.top_ids[-1], self.bottom_ids[-1], self._edge_rest(self.top_ids[-1], self.bottom_ids[-1]), 0.54, "cross"))
        return edges

    def hinge_world_positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        p0 = np.array([0.0, 0.0])
        theta1 = q[0]
        p1 = p0 + self.link_lengths[0] * np.array([math.cos(theta1), math.sin(theta1)])
        theta2 = q[0] + q[1]
        p2 = p1 + self.link_lengths[1] * np.array([math.cos(theta2), math.sin(theta2)])
        theta3 = q[0] + q[1] + q[2]
        p3 = p2 + self.link_lengths[2] * np.array([math.cos(theta3), math.sin(theta3)])
        return np.vstack((p0, p1, p2, p3))

    def map_points_to_outer(self, points: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Corrected v4b terminal mapping: s == sleeve_length stays pre-DIP."""
        q = np.asarray(q, dtype=float)
        points = np.asarray(points, dtype=float)
        hinges_full = np.array([self.hinge_positions[0], self.hinge_positions[1], self.sleeve_length])
        hinge_world = self.hinge_world_positions(q)
        out = np.zeros_like(points)

        for idx, p in enumerate(points):
            s, y = p
            if s >= self.sleeve_length - 1e-9:
                seg = 2
            else:
                seg = int(np.searchsorted(hinges_full, s, side="right"))

            base_s = 0.0 if seg == 0 else hinges_full[seg - 1]
            if seg == 0:
                base_world = np.array([0.0, 0.0])
                theta = 0.0
            else:
                base_world = hinge_world[seg - 1]
                theta = float(np.sum(q[:seg]))
            out[idx] = base_world + rotation(theta) @ np.array([s - base_s, y])
        return out

    def anchors(self, q: np.ndarray) -> np.ndarray:
        return self.map_points_to_outer(self.local_nodes, q)

    def contact_node_ids(self, target: str = "all") -> np.ndarray:
        s = self.s_values
        if target == "middle":
            mask = (s >= self.link_lengths[0] + 0.10 * self.link_lengths[1]) & (s <= self.sleeve_length - 1.0)
        elif target == "proximal":
            mask = (s >= 0.15 * self.link_lengths[0]) & (s <= 0.92 * self.link_lengths[0])
        else:
            mask = (s >= 2.0) & (s <= self.sleeve_length - 1e-6)
        return self.bottom_ids[mask]

    def distal_cap_polygon(self, q: np.ndarray) -> np.ndarray:
        hinges = self.hinge_world_positions(q)
        dip, tip = hinges[2], hinges[3]
        axis = unit(tip - dip)
        normal = np.array([-axis[1], axis[0]])
        root_half = self.distal_half_width
        tip_half = 0.82 * self.distal_half_width
        return np.vstack((
            dip + root_half * normal,
            tip + tip_half * normal,
            tip - tip_half * normal,
            dip - root_half * normal,
        ))

    def energy_and_grad(
        self,
        x_flat: np.ndarray,
        q: np.ndarray,
        anchors: np.ndarray,
        object_geom=None,
        contact_node_ids: np.ndarray | None = None,
        include_contact: bool = True,
        k_contact: float | None = None,
    ) -> tuple[float, np.ndarray]:
        x = x_flat.reshape((-1, 2))
        grad = np.zeros_like(x)
        spring_energy = 0.0

        for edge in self.edges:
            d = x[edge.i] - x[edge.j]
            length = float(np.linalg.norm(d))
            if length < 1e-12:
                continue
            strain_len = length - edge.rest
            e = 0.5 * edge.k * strain_len * strain_len
            g = edge.k * strain_len * d / length
            spring_energy += e
            grad[edge.i] += g
            grad[edge.j] -= g

        delta = x - anchors
        anchor_energy = 0.5 * self.k_anchor * float(np.sum(delta * delta))
        grad += self.k_anchor * delta

        contact_energy = 0.0
        if include_contact and object_geom is not None and contact_node_ids is not None:
            ids = np.asarray(contact_node_ids, dtype=int)
            phi = object_geom.signed_distance(x[ids])
            penetration = np.maximum(0.0, -phi)
            active = penetration > 0.0
            if np.any(active):
                kc = self.k_contact_default if k_contact is None else float(k_contact)
                active_ids = ids[active]
                normals = object_geom.normal(x[active_ids])
                contact_energy = 0.5 * kc * float(np.sum(penetration[active] ** 2))
                grad[active_ids] += -kc * penetration[active, None] * normals

        total = spring_energy + anchor_energy + contact_energy
        return total, grad.ravel()

    def energy_components(
        self,
        x: np.ndarray,
        q: np.ndarray,
        anchors: np.ndarray,
        object_geom=None,
        contact_node_ids: np.ndarray | None = None,
        include_contact: bool = True,
        k_contact: float | None = None,
    ) -> dict[str, float]:
        x = np.asarray(x, dtype=float)
        spring_energy = 0.0
        for edge in self.edges:
            d = x[edge.i] - x[edge.j]
            length = float(np.linalg.norm(d))
            spring_energy += 0.5 * edge.k * (length - edge.rest) ** 2
        delta = x - anchors
        anchor_energy = 0.5 * self.k_anchor * float(np.sum(delta * delta))
        contact_energy = 0.0
        if include_contact and object_geom is not None and contact_node_ids is not None:
            ids = np.asarray(contact_node_ids, dtype=int)
            phi = object_geom.signed_distance(x[ids])
            penetration = np.maximum(0.0, -phi)
            kc = self.k_contact_default if k_contact is None else float(k_contact)
            contact_energy = 0.5 * kc * float(np.sum(penetration ** 2))
        return {
            "spring_energy": spring_energy,
            "anchor_energy": anchor_energy,
            "contact_energy": contact_energy,
            "total_energy": spring_energy + anchor_energy + contact_energy,
        }

    def solve(
        self,
        q: np.ndarray,
        object_geom=None,
        contact_node_ids: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        include_contact: bool = True,
        k_contact: float | None = None,
        maxiter: int = 900,
    ):
        anchors = self.anchors(q)
        x0 = anchors if x_init is None else np.asarray(x_init, dtype=float)

        def fun(flat):
            return self.energy_and_grad(
                flat,
                q,
                anchors,
                object_geom=object_geom,
                contact_node_ids=contact_node_ids,
                include_contact=include_contact,
                k_contact=k_contact,
            )

        res = minimize(
            fun,
            x0.ravel(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
        )
        return res.x.reshape((-1, 2)), anchors, res

    def solve_with_object(
        self,
        q: np.ndarray,
        object_geom,
        x_init: np.ndarray | None = None,
        contact_node_ids: np.ndarray | None = None,
    ):
        if contact_node_ids is None:
            contact_node_ids = self.contact_node_ids("all")
        x, anchors, _ = self.solve(q, x_init=x_init, include_contact=False)

        target_center = np.asarray(object_geom.center, dtype=float)
        contact_anchor = anchors[np.asarray(contact_node_ids, dtype=int)]
        nearest_idx = int(np.argmin(object_geom.signed_distance(contact_anchor)))
        p_near = contact_anchor[nearest_idx]
        direction = unit(target_center - p_near)
        start_center = target_center + 7.5 * direction

        res = None
        for alpha in np.linspace(0.0, 1.0, 9):
            center = (1.0 - alpha) * start_center + alpha * target_center
            obj_alpha = object_geom.with_center(center)
            kc = 280.0 + alpha * (self.k_contact_default - 280.0)
            x, anchors, res = self.solve(
                q,
                object_geom=obj_alpha,
                contact_node_ids=contact_node_ids,
                x_init=x,
                include_contact=True,
                k_contact=kc,
                maxiter=700,
            )

        # Final high-stiffness polish.
        x, anchors, res = self.solve(
            q,
            object_geom=object_geom,
            contact_node_ids=contact_node_ids,
            x_init=x,
            include_contact=True,
            k_contact=self.k_contact_default * 2.5,
            maxiter=1100,
        )
        return x, anchors, res

    def edge_strains(self, x: np.ndarray) -> dict[int, float]:
        strains = {}
        for idx, edge in enumerate(self.edges):
            length = float(np.linalg.norm(x[edge.i] - x[edge.j]))
            strains[idx] = (length - edge.rest) / max(edge.rest, 1e-12)
        return strains

    def metrics(
        self,
        x: np.ndarray,
        q: np.ndarray,
        anchors: np.ndarray,
        object_geom=None,
        contact_node_ids: np.ndarray | None = None,
        case: str = "",
        object_type: str = "none",
        success: bool = True,
    ) -> dict[str, float | str | bool | int]:
        components = self.energy_components(
            x,
            q,
            anchors,
            object_geom=object_geom,
            contact_node_ids=contact_node_ids,
            include_contact=object_geom is not None,
            k_contact=self.k_contact_default * 2.5,
        )
        strains = self.edge_strains(x)
        all_abs = [abs(v) for v in strains.values()]
        hinge_abs = [
            abs(strains[idx])
            for idx, edge in enumerate(self.edges)
            if edge.kind == "diag_hinge"
        ]

        active_nodes = 0
        max_pen_contact = 0.0
        if object_geom is not None and contact_node_ids is not None:
            phi = object_geom.signed_distance(x[np.asarray(contact_node_ids, dtype=int)])
            penetration = np.maximum(0.0, -phi)
            max_pen_contact = float(np.max(penetration)) if len(penetration) else 0.0
            active_nodes = int(np.sum(phi < 0.05))

        max_pen_cap = 0.0
        if object_geom is not None:
            cap_pts = sample_polygon_boundary(self.distal_cap_polygon(q), 96)
            cap_phi = object_geom.signed_distance(cap_pts)
            max_pen_cap = float(np.max(np.maximum(0.0, -cap_phi)))

        return {
            "case": case,
            "mode": self.mode,
            "object_type": object_type,
            **components,
            "active_contact_nodes": active_nodes,
            "max_penetration_contact_nodes": max_pen_contact,
            "max_penetration_distal_cap": max_pen_cap,
            "max_node_displacement_from_anchor": float(np.max(np.linalg.norm(x - anchors, axis=1))),
            "peak_abs_strain_all": float(max(all_abs) if all_abs else 0.0),
            "peak_hinge_crossing_diag_strain": float(max(hinge_abs) if hinge_abs else 0.0),
            "success": bool(success),
        }


class HingeCompatibleSleeveV4b(HingeCompatibleSleeveV4):
    """Alias class retaining the corrected terminal mapping requested for v4b."""


def sample_polygon_boundary(poly: np.ndarray, n_per_edge: int = 32) -> np.ndarray:
    pts = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        t = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        pts.append((1.0 - t[:, None]) * a[None, :] + t[:, None] * b[None, :])
    return np.vstack(pts)


def object_bounds(obj) -> np.ndarray:
    if isinstance(obj, CircleObject):
        c = obj.center
        r = obj.radius
        return np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]])
    if isinstance(obj, BoxObject):
        v = obj.vertices()
        return np.vstack((np.min(v, axis=0), np.max(v, axis=0)))
    raise TypeError(type(obj))


def place_object_in_grasp(
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    kind: str = "circle",
    initial_penetration: float = 1.1,
):
    anchors = model.anchors(q)
    top = anchors[model.top_ids]
    bottom = anchors[model.bottom_ids]
    s = model.s_values

    if kind == "circle":
        radius = 8.6
        target_s = model.link_lengths[0] + 0.45 * model.link_lengths[1]
    elif kind == "flat_box":
        radius = None
        target_s = model.link_lengths[0] + 0.50 * model.link_lengths[1]
    else:
        radius = None
        target_s = model.link_lengths[0] + 0.34 * model.link_lengths[1]

    idx = int(np.argmin(np.abs(s - target_s)))
    p_contact = bottom[idx]
    palmar = unit(bottom[idx] - top[idx])
    tangent = unit(bottom[min(idx + 1, model.N)] - bottom[max(idx - 1, 0)], fallback=np.array([1.0, 0.0]))

    if kind == "circle":
        center = p_contact + palmar * (radius - initial_penetration)
        obj = CircleObject(center, radius, label="circle")
    elif kind == "flat_box":
        width, height = 15.8, 7.2
        center = p_contact + palmar * (0.5 * height - initial_penetration)
        obj = BoxObject(center, width, height, rotation=math.atan2(tangent[1], tangent[0]), label="flat box")
    elif kind == "edge_box":
        width, height = 8.4, 12.6
        center = p_contact + palmar * (0.5 * height - initial_penetration)
        obj = BoxObject(center, width, height, rotation=math.atan2(tangent[1], tangent[0]) + 0.16, label="edge box")
    else:
        raise ValueError(f"unknown object kind: {kind}")

    # Keep the rigid distal cap from penetrating by backing the object out along
    # the palmar direction. The sleeve remains the adaptive surface.
    for _ in range(14):
        cap_pts = sample_polygon_boundary(model.distal_cap_polygon(q), 48)
        penetration = np.max(np.maximum(0.0, -obj.signed_distance(cap_pts)))
        if penetration <= 1e-4:
            break
        obj = obj.with_center(obj.center + palmar * (penetration + 0.8))
    return obj


def active_contact_ids(model: HingeCompatibleSleeveV4, x: np.ndarray, obj, ids: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    phi = obj.signed_distance(x[ids])
    return ids[phi < threshold]


def draw_skeleton(ax, model: HingeCompatibleSleeveV4, q: np.ndarray, alpha: float = 0.58, zorder: float = 2.5) -> None:
    center_y = 0.5 * (model.pad_inner_offset + model.pad_outer_offset)
    base_chain = model.map_points_to_outer(
        np.array([
            [0.0, center_y],
            [model.link_lengths[0], center_y],
            [model.sleeve_length, center_y],
        ]),
        q,
    )
    distal_poly = distal_trainable_interface_polygon(model, q)
    distal_centerline = 0.5 * (distal_poly[:8] + distal_poly[8:][::-1])
    hinges = np.vstack((base_chain, distal_centerline[-1]))
    ax.plot(
        hinges[:, 0],
        hinges[:, 1],
        "-",
        color="#222222",
        linewidth=1.35,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.scatter(
        hinges[:, 0],
        hinges[:, 1],
        s=[20, 18, 18, 14],
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.75,
        alpha=alpha,
        zorder=zorder + 0.2,
    )


def linkage_display_points(model: HingeCompatibleSleeveV4, q: np.ndarray) -> dict[str, np.ndarray]:
    """Construct a readable closed-linkage overlay from the prescribed gross pose."""
    O, A, B, _ = model.hinge_world_positions(q)
    prox_axis = unit(A - O)
    prox_dorsal = np.array([-prox_axis[1], prox_axis[0]])
    mid_axis = unit(B - A)
    mid_dorsal = np.array([-mid_axis[1], mid_axis[0]])
    mid_palmar = -mid_dorsal

    # Compact display geometry: the colored links follow the gross pose, while
    # the distal B-C link is intentionally routed toward the palmar side.
    C = B + 0.52 * model.link_lengths[2] * mid_axis + 7.2 * mid_palmar
    Ap = O + 0.60 * (A - O) + 3.8 * prox_dorsal
    Bp = A + 0.52 * (B - A) + 3.6 * mid_dorsal
    Cp = B + 0.46 * (C - B) + 2.8 * mid_palmar
    Op = O + 0.20 * (A - O) + 6.0 * prox_dorsal

    return {"O": O, "Op": Op, "A": A, "Ap": Ap, "B": B, "Bp": Bp, "C": C, "Cp": Cp}


def linkage_points_array(model: HingeCompatibleSleeveV4, q: np.ndarray) -> np.ndarray:
    pts = linkage_display_points(model, q)
    return np.vstack([pts[key] for key in ("O", "Op", "A", "Ap", "B", "Bp", "C", "Cp")])


def draw_deterministic_linkage(
    ax,
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    alpha: float = 1.0,
    zorder: float = 2.6,
) -> None:
    """Draw rigid linkage bodies and couplers so the gross pose reads as linkage."""
    pts = linkage_display_points(model, q)
    body_specs = [
        ("proximal", [pts["O"], pts["A"], pts["Ap"]], "#d62728"),
        ("middle", [pts["A"], pts["B"], pts["Bp"]], "#1f77b4"),
        ("distal", [pts["B"], pts["C"], pts["Cp"]], "#2ca02c"),
    ]
    for _, tri, color in body_specs:
        tri_arr = np.asarray(tri)
        ax.add_patch(Polygon(
            tri_arr,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.10 * alpha,
            zorder=zorder,
        ))
        closed = np.vstack((tri_arr, tri_arr[0]))
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=color,
            linewidth=1.25,
            alpha=0.58 * alpha,
            zorder=zorder + 0.1,
        )

    for a, b in (("Op", "Bp"), ("Ap", "Cp")):
        seg = np.vstack((pts[a], pts[b]))
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color="#171717",
            linewidth=1.45,
            alpha=0.72 * alpha,
            solid_capstyle="round",
            zorder=zorder + 0.35,
        )

    for a, b, color in (
        ("O", "A", "#d62728"),
        ("A", "B", "#1f77b4"),
        ("B", "C", "#2ca02c"),
    ):
        seg = np.vstack((pts[a], pts[b]))
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=color,
            linewidth=2.4,
            alpha=0.95 * alpha,
            solid_capstyle="round",
            zorder=zorder + 0.5,
        )

    primary_points = np.vstack((pts["O"], pts["A"], pts["B"], pts["C"]))
    primary_colors = ["#d62728", "#1f77b4", "#1f77b4", "#2ca02c"]
    ax.scatter(
        primary_points[:, 0],
        primary_points[:, 1],
        s=[48, 42, 42, 36],
        facecolor=primary_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95 * alpha,
        zorder=zorder + 0.78,
    )

    aux_points = np.vstack((pts["Op"], pts["Ap"], pts["Bp"], pts["Cp"]))
    ax.scatter(
        aux_points[:, 0],
        aux_points[:, 1],
        s=[24, 22, 22, 22],
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.75,
        alpha=0.92 * alpha,
        zorder=zorder + 0.75,
    )


def draw_distal_cap(ax, model: HingeCompatibleSleeveV4, q: np.ndarray, alpha: float = 1.0, zorder: float = 4.2) -> None:
    """Keep distal-cap geometry for diagnostics, but do not draw a visual box."""
    return


def distal_trainable_interface_polygon(model: HingeCompatibleSleeveV4, q: np.ndarray) -> np.ndarray:
    """Smooth distal contact interface, visually distinct from fin-ray lattice."""
    theta_middle = float(np.sum(q[:2]))
    middle_axis = rotation(theta_middle) @ np.array([1.0, 0.0])
    palmar = rotation(theta_middle) @ np.array([0.0, -1.0])
    distal_axis = unit(0.32 * middle_axis + 0.95 * palmar)
    width_axis = np.array([-distal_axis[1], distal_axis[0]])

    root = model.map_points_to_outer(
        np.array([[model.sleeve_length, 0.5 * (model.pad_inner_offset + model.pad_outer_offset)]], dtype=float),
        q,
    )[0]
    root = root + 0.7 * distal_axis

    stations = np.linspace(0.0, 1.0, 8)
    length = 13.2
    centerline = root[None, :] + stations[:, None] * length * distal_axis[None, :]
    taper = np.linspace(1.15, 0.82, len(stations))
    half_width = 2.2 * taper
    side_a = centerline + half_width[:, None] * width_axis[None, :]
    side_b = centerline - half_width[:, None] * width_axis[None, :]
    return np.vstack((side_a, side_b[::-1]))


def draw_trainable_distal_interface(
    ax,
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    alpha: float = 0.42,
    zorder: float = 2.8,
) -> None:
    poly = distal_trainable_interface_polygon(model, q)
    ax.add_patch(Polygon(
        poly,
        closed=True,
        facecolor="#8e6bbd",
        edgecolor="#5e3c99",
        linewidth=0.85,
        alpha=alpha,
        zorder=zorder,
    ))

    control = poly[:8][[1, 3, 5, 6]]
    ax.scatter(
        control[:, 0],
        control[:, 1],
        s=10,
        marker="s",
        facecolor="#8e6bbd",
        edgecolor="white",
        linewidth=0.45,
        alpha=min(1.0, alpha + 0.26),
        zorder=zorder + 0.15,
    )


def draw_sleeve(
    ax,
    model: HingeCompatibleSleeveV4,
    x: np.ndarray,
    color: str = "#1b9e9a",
    alpha: float = 0.95,
    linewidth: float = 1.0,
    zorder: float = 3.0,
    strain_reference: np.ndarray | None = None,
) -> None:
    pad_poly = np.vstack((x[model.top_ids], x[model.bottom_ids[::-1]]))
    ax.add_patch(Polygon(
        pad_poly,
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=0.045 * alpha,
        zorder=zorder - 0.2,
    ))

    segments = []
    colors = []
    widths = []
    for edge in model.edges:
        segments.append([x[edge.i], x[edge.j]])
        if strain_reference is None:
            colors.append(color)
            widths.append(linewidth if edge.kind != "diag_hinge" else linewidth * 0.82)
        else:
            length = float(np.linalg.norm(x[edge.i] - x[edge.j]))
            ref_len = float(np.linalg.norm(strain_reference[edge.i] - strain_reference[edge.j]))
            delta = abs((length - edge.rest) / edge.rest - (ref_len - edge.rest) / edge.rest)
            t = min(1.0, delta / 0.10)
            base = np.array([27, 158, 154], dtype=float) / 255.0
            hot = np.array([226, 97, 53], dtype=float) / 255.0
            colors.append(tuple((1.0 - t) * base + t * hot))
            widths.append(linewidth * (0.95 + 1.2 * t))

    lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    ax.scatter(
        x[:, 0],
        x[:, 1],
        s=3.4,
        facecolor="white",
        edgecolor=color,
        linewidth=0.24,
        alpha=0.78 * alpha,
        zorder=zorder + 0.05,
    )


def draw_reference_sleeve(ax, model: HingeCompatibleSleeveV4, anchors: np.ndarray, alpha: float = 0.32) -> None:
    draw_sleeve(ax, model, anchors, color="#9a9a9a", alpha=alpha, linewidth=0.66, zorder=1.4)


def draw_contact_normals(ax, obj, x: np.ndarray, active_ids: np.ndarray, scale: float = 3.0) -> None:
    if len(active_ids) == 0:
        return
    chosen = active_ids[:: max(1, len(active_ids) // 5)]
    pts = x[chosen]
    normals = obj.normal(pts)
    for p, n in zip(pts, normals):
        ax.annotate(
            "",
            xy=p + scale * n,
            xytext=p,
            arrowprops={"arrowstyle": "->", "color": "#d9481e", "lw": 1.0, "shrinkA": 0, "shrinkB": 0},
            zorder=6,
        )


def annotate_panel_metrics(ax, metrics: dict) -> None:
    ax.text(
        0.02,
        0.04,
        f"max penetration = {metrics['max_penetration_contact_nodes']:.2e}\n"
        f"active nodes = {metrics['active_contact_nodes']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.86},
        zorder=10,
    )


def clean_axis(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_limits_from_points(ax, point_sets: list[np.ndarray], pad: float = 7.0) -> None:
    pts = np.vstack([np.asarray(p).reshape((-1, 2)) for p in point_sets if len(np.asarray(p).reshape((-1, 2)))])
    mn = np.min(pts, axis=0) - pad
    mx = np.max(pts, axis=0) + pad
    span = mx - mn
    if span[0] < span[1]:
        extra = 0.5 * (span[1] - span[0])
        mn[0] -= extra
        mx[0] += extra
    else:
        extra = 0.5 * (span[0] - span[1])
        mn[1] -= extra
        mx[1] += extra
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])


def draw_case_panel(
    ax,
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    unloaded_x: np.ndarray,
    anchors: np.ndarray,
    case: dict,
    title: str,
    panel_label: str,
    show_object: bool = True,
    show_normals: bool = True,
) -> None:
    clean_axis(ax)
    ax.set_title(title, pad=4)
    ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=11)
    draw_reference_sleeve(ax, model, anchors, alpha=0.18)
    draw_skeleton(ax, model, q)
    draw_trainable_distal_interface(ax, model, q, alpha=0.30)
    draw_distal_cap(ax, model, q)

    if show_object and case.get("object") is not None:
        case["object"].draw(ax)
        draw_sleeve(ax, model, case["x"], color="#138f8a", alpha=0.66, linewidth=0.96, zorder=3.4, strain_reference=unloaded_x)
        active = case["active_ids"]
        if len(active):
            ax.scatter(
                case["x"][active, 0],
                case["x"][active, 1],
                s=20,
                facecolor="#d9481e",
                edgecolor="white",
                linewidth=0.42,
                zorder=6.1,
            )
        if show_normals:
            draw_contact_normals(ax, case["object"], case["x"], active)
        annotate_panel_metrics(ax, case["metrics"])
    else:
        draw_sleeve(ax, model, unloaded_x, color="#138f8a", alpha=0.60, linewidth=0.96, zorder=3.4)
        ax.text(
            0.50,
            0.05,
            "low unloaded internal strain",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef7f6", "edgecolor": "#9bc8c4", "linewidth": 0.6},
        )


def make_grasp_figure(
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    unloaded_x: np.ndarray,
    anchors: np.ndarray,
    cases: list[dict],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), constrained_layout=True)

    draw_case_panel(
        axes[0],
        model,
        q,
        unloaded_x,
        anchors,
        {"object": None, "x": unloaded_x},
        "Unloaded deterministic flexion",
        "(a)",
        show_object=False,
    )
    draw_case_panel(
        axes[1],
        model,
        q,
        unloaded_x,
        anchors,
        cases[0],
        "Object contact activates pad deformation",
        "(b)",
    )

    ax = axes[2]
    clean_axis(ax)
    ax.set_title("Same gross pose, different object", pad=4)
    ax.text(0.02, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=11)
    draw_reference_sleeve(ax, model, anchors, alpha=0.16)
    draw_skeleton(ax, model, q, alpha=0.46)
    draw_trainable_distal_interface(ax, model, q, alpha=0.24)
    draw_distal_cap(ax, model, q)
    colors = ["#138f8a", "#4c78a8", "#7a9d35"]
    for case, color in zip(cases, colors):
        case["object"].draw(ax, alpha=0.36, zorder=1.1)
        draw_sleeve(ax, model, case["x"], color=color, alpha=0.42, linewidth=0.86, zorder=3.0, strain_reference=unloaded_x)
        active = case["active_ids"]
        if len(active):
            ax.scatter(case["x"][active, 0], case["x"][active, 1], s=11, color="#d9481e", alpha=0.72, zorder=6)
    ax.text(
        0.50,
        0.05,
        "same q; local contact interfaces change",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88},
    )

    point_sets = [anchors, unloaded_x, distal_trainable_interface_polygon(model, q)]
    for case in cases:
        point_sets.extend([case["x"], object_bounds(case["object"])])
    for ax in axes:
        set_limits_from_points(ax, point_sets, pad=7.5)

    handles = [
        plt.Line2D([0], [0], color="#9a9a9a", lw=1.4, label="deterministic reference"),
        plt.Line2D([0], [0], color="#222222", lw=1.35, alpha=0.58, label="kinematic serial chain"),
        plt.Line2D([0], [0], color="#138f8a", lw=1.8, label="palmar compliant pad"),
        plt.Line2D([0], [0], color="#8e6bbd", lw=5.0, alpha=0.42, label="trainable distal interface"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9481e", markeredgecolor="white", label="active contact nodes"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=5,
        frameon=True,
        framealpha=0.94,
        handlelength=2.0,
    )

    fig.savefig(out_dir / "clear_contact_interface_grasp.png", bbox_inches="tight")
    fig.savefig(out_dir / "clear_contact_interface_grasp.pdf", bbox_inches="tight")
    plt.close(fig)


def make_object_cases_figure(
    model: HingeCompatibleSleeveV4,
    q: np.ndarray,
    unloaded_x: np.ndarray,
    anchors: np.ndarray,
    cases: list[dict],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(12.0, 4.2), constrained_layout=True)
    point_sets = [anchors, unloaded_x, distal_trainable_interface_polygon(model, q)]
    for case in cases:
        point_sets.extend([case["x"], object_bounds(case["object"])])

    for idx, (ax, case) in enumerate(zip(axes, cases)):
        draw_case_panel(
            ax,
            model,
            q,
            unloaded_x,
            anchors,
            case,
            case["title"],
            f"({chr(ord('a') + idx)})",
            show_object=True,
            show_normals=True,
        )
        set_limits_from_points(ax, point_sets, pad=7.5)

    fig.savefig(out_dir / "clear_contact_interface_object_cases.png", bbox_inches="tight")
    fig.savefig(out_dir / "clear_contact_interface_object_cases.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model(ax, model: HingeCompatibleSleeveV4, x: np.ndarray, q: np.ndarray, anchors: np.ndarray, title: str) -> None:
    clean_axis(ax)
    ax.set_title(title)
    draw_reference_sleeve(ax, model, anchors, alpha=0.18)
    draw_sleeve(ax, model, x, color="#138f8a", alpha=0.60, linewidth=0.90, zorder=3)
    draw_trainable_distal_interface(ax, model, q, alpha=0.30)
    draw_distal_cap(ax, model, q)
    set_limits_from_points(ax, [anchors, x, distal_trainable_interface_polygon(model, q)], pad=7.0)


def make_v4_unloaded_comparisons(out_dir: Path) -> list[dict]:
    q_demo = np.deg2rad([-25.0, -65.0, -35.0])
    naive = HingeCompatibleSleeveV4b(mode="naive", n_segments=48)
    routed = HingeCompatibleSleeveV4b(mode="routed", n_segments=48)

    rows: list[dict] = []
    x_naive, a_naive, res_naive = naive.solve(q_demo, include_contact=False)
    x_routed, a_routed, res_routed = routed.solve(q_demo, include_contact=False)
    rows.append(naive.metrics(x_naive, q_demo, a_naive, case="naive_unloaded", object_type="none", success=res_naive.success))
    rows.append(routed.metrics(x_routed, q_demo, a_routed, case="routed_unloaded", object_type="none", success=res_routed.success))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    plot_model(axes[0], naive, x_naive, q_demo, a_naive, "v4b naive bonded sleeve\nunloaded flexion")
    plot_model(axes[1], routed, x_routed, q_demo, a_routed, "v4b neutral-axis routed sleeve\nunloaded flexion")
    fig.savefig(out_dir / "v4_naive_vs_routed_unloaded.png", bbox_inches="tight")
    fig.savefig(out_dir / "v4b_naive_vs_routed_unloaded.png", bbox_inches="tight")
    plt.close(fig)

    sweep_rows = []
    xprev_naive = None
    xprev_routed = None
    for pip_deg in np.linspace(0.0, -85.0, 12):
        q = np.deg2rad([-25.0, pip_deg, -35.0])
        for model, xprev in ((naive, xprev_naive), (routed, xprev_routed)):
            x, a, res = model.solve(q, x_init=xprev, include_contact=False)
            met = model.metrics(x, q, a, case=f"{model.mode}_pip_{pip_deg:.1f}", object_type="none", success=res.success)
            met["pip_deg"] = float(pip_deg)
            sweep_rows.append(met)
            if model.mode == "naive":
                xprev_naive = x
            else:
                xprev_routed = x

    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    for mode in ("naive", "routed"):
        data = [r for r in sweep_rows if r["mode"] == mode]
        ax.plot([-r["pip_deg"] for r in data], [r["spring_energy"] for r in data], marker="o", label=mode)
    ax.set_title("v4b unloaded flexion: internal spring energy")
    ax.set_xlabel("PIP flexion magnitude [deg]")
    ax.set_ylabel("spring energy [arb.]")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend()
    fig.savefig(out_dir / "v4_unloaded_energy_vs_flexion.png", bbox_inches="tight")
    fig.savefig(out_dir / "v4b_unloaded_energy_vs_flexion.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    for mode in ("naive", "routed"):
        data = [r for r in sweep_rows if r["mode"] == mode]
        ax.plot([-r["pip_deg"] for r in data], [r["peak_hinge_crossing_diag_strain"] for r in data], marker="o", label=mode)
    ax.set_title("v4b unloaded flexion: hinge-crossing rib strain")
    ax.set_xlabel("PIP flexion magnitude [deg]")
    ax.set_ylabel("peak absolute strain")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend()
    fig.savefig(out_dir / "v4_hinge_crossing_rib_strain_vs_flexion.png", bbox_inches="tight")
    fig.savefig(out_dir / "v4b_hinge_crossing_rib_strain_vs_flexion.png", bbox_inches="tight")
    plt.close(fig)

    write_metrics_csv(out_dir / "v4b_flexion_sweep_metrics.csv", sweep_rows)
    return rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_non_grasp_outputs(out_dir: Path) -> None:
    """Keep only the final grasp figure outputs from this script."""
    keep = {
        "clear_contact_interface_grasp.png",
        "clear_contact_interface_grasp.pdf",
    }
    generated_patterns = [
        "clear_contact_interface_object_cases.*",
        "clear_contact_interface_metrics.csv",
        "v4_*.png",
        "v4b_*.png",
        "v4b_*.csv",
    ]
    for pattern in generated_patterns:
        for path in out_dir.glob(pattern):
            if path.name not in keep and path.is_file():
                path.unlink()


def build_contact_cases(model: HingeCompatibleSleeveV4, q_demo: np.ndarray, unloaded_x: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    anchors = model.anchors(q_demo)
    cases: list[dict] = []
    for kind, title in (
        ("circle", "Round object"),
        ("flat_box", "Flat object"),
        ("edge_box", "Edge-like object"),
    ):
        obj = place_object_in_grasp(model, q_demo, kind=kind, initial_penetration=1.25)
        contact_ids = model.contact_node_ids("all")
        x, anchors, res = model.solve_with_object(q_demo, obj, x_init=unloaded_x, contact_node_ids=contact_ids)
        metrics = model.metrics(
            x,
            q_demo,
            anchors,
            object_geom=obj,
            contact_node_ids=contact_ids,
            case=f"routed_contact_{kind}",
            object_type=kind,
            success=res.success,
        )
        cases.append({
            "kind": kind,
            "title": title,
            "object": obj,
            "x": x,
            "anchors": anchors,
            "result": res,
            "metrics": metrics,
            "active_ids": active_contact_ids(model, x, obj, contact_ids),
        })
    return anchors, cases


def main() -> None:
    set_research_style()
    out_dir = prepare_output_dir()
    cleanup_non_grasp_outputs(out_dir)

    q_demo = np.deg2rad([-25.0, -65.0, -35.0])
    model = HingeCompatibleSleeveV4b(mode="routed", n_segments=48)
    unloaded_x, _, _ = model.solve(q_demo, include_contact=False)

    anchors, cases = build_contact_cases(model, q_demo, unloaded_x)

    make_grasp_figure(model, q_demo, unloaded_x, anchors, cases, out_dir)
    cleanup_non_grasp_outputs(out_dir)

    print("Saved outputs to", out_dir)
    for path in sorted(out_dir.glob("clear_contact_interface_grasp.*")):
        print(" ", path)
    print("\nContact diagnostics:")
    for case in cases:
        m = case["metrics"]
        print(
            f"  {case['kind']}: max penetration contact={m['max_penetration_contact_nodes']:.3e}, "
            f"distal cap={m['max_penetration_distal_cap']:.3e}, "
            f"active nodes={m['active_contact_nodes']}, success={m['success']}"
        )


if __name__ == "__main__":
    main()
