"""Generic 2D contact geometry utilities.

The objects here are intentionally model-agnostic. A mechanism provides sampled
points with clearance radii, and this module evaluates no-penetration margins
against analytic object signed-distance functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray


def unit(v: Array) -> Array:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / norm


class ContactObject2D:
    """Protocol-like base for analytic 2D contact objects."""

    def signed_distance(self, points: Array) -> Array:
        raise NotImplementedError

    def normal(self, points: Array) -> Array:
        raise NotImplementedError

    def closest_point(self, points: Array) -> Array:
        points_arr = np.asarray(points, dtype=float).reshape(-1, 2)
        phi = self.signed_distance(points_arr)
        return points_arr - phi[:, None] * self.normal(points_arr)


@dataclass(frozen=True)
class CircleObject(ContactObject2D):
    """Circle contact object with positive-outside signed distance."""

    center: Sequence[float]
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(2)
        radius = float(self.radius)
        if radius <= 0.0:
            raise ValueError("CircleObject radius must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    def signed_distance(self, points: Array) -> Array:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.linalg.norm(pts - self.center[None, :], axis=1) - self.radius

    def normal(self, points: Array) -> Array:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        delta = pts - self.center[None, :]
        norm = np.linalg.norm(delta, axis=1)
        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        normals = delta / safe_norm[:, None]
        normals[norm <= 1e-12] = np.array([1.0, 0.0])
        return normals

    def closest_point(self, points: Array) -> Array:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return self.center[None, :] + self.radius * self.normal(pts)


@dataclass(frozen=True)
class BoxObject(ContactObject2D):
    """Rotated rectangle signed-distance object."""

    center: Sequence[float]
    width: float
    height: float
    rotation: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(2)
        width = float(self.width)
        height = float(self.height)
        if width <= 0.0 or height <= 0.0:
            raise ValueError("BoxObject width and height must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "rotation", float(self.rotation))

    @property
    def _rot(self) -> Array:
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _to_local(self, points: Array) -> Array:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return (self._rot.T @ (pts - self.center[None, :]).T).T

    def _to_world_vec(self, vectors: Array) -> Array:
        return (self._rot @ np.asarray(vectors, dtype=float).reshape(-1, 2).T).T

    def signed_distance(self, points: Array) -> Array:
        local = self._to_local(points)
        half = np.array([0.5 * self.width, 0.5 * self.height], dtype=float)
        q = np.abs(local) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        return outside + inside

    def normal(self, points: Array) -> Array:
        local = self._to_local(points)
        half = np.array([0.5 * self.width, 0.5 * self.height], dtype=float)
        q = np.abs(local) - half[None, :]
        normals = np.zeros_like(local)

        outside = np.any(q > 0.0, axis=1)
        if np.any(outside):
            vec = np.sign(local[outside]) * np.maximum(q[outside], 0.0)
            norms = np.linalg.norm(vec, axis=1)
            safe = np.where(norms > 1e-12, norms, 1.0)
            normals[outside] = vec / safe[:, None]

        inside = ~outside
        if np.any(inside):
            choose_x = q[inside, 0] >= q[inside, 1]
            inside_normals = np.zeros((int(np.sum(inside)), 2), dtype=float)
            inside_normals[choose_x, 0] = np.sign(local[inside][choose_x, 0])
            inside_normals[~choose_x, 1] = np.sign(local[inside][~choose_x, 1])
            inside_normals[np.linalg.norm(inside_normals, axis=1) < 1e-12] = np.array([1.0, 0.0])
            normals[inside] = inside_normals

        return self._to_world_vec(normals)


@dataclass(frozen=True)
class ContactSampleSet:
    """Sampled mechanism geometry for no-penetration checks."""

    points: Array
    radii: Array
    owner_ids: Array
    owner_names: tuple[str, ...]
    fractions: Array | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        count = len(points)
        radii = np.asarray(self.radii, dtype=float).reshape(-1)
        owner_ids = np.asarray(self.owner_ids, dtype=int).reshape(-1)
        if radii.size != count:
            raise ValueError("radii must match sampled points")
        if owner_ids.size != count:
            raise ValueError("owner_ids must match sampled points")
        if len(self.owner_names) != count:
            raise ValueError("owner_names must match sampled points")
        if self.fractions is None:
            fractions = np.full(count, np.nan, dtype=float)
        else:
            fractions = np.asarray(self.fractions, dtype=float).reshape(-1)
            if fractions.size != count:
                raise ValueError("fractions must match sampled points")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "owner_ids", owner_ids)
        object.__setattr__(self, "owner_names", tuple(str(v) for v in self.owner_names))
        object.__setattr__(self, "fractions", fractions)

    def margins(self, object_geom: ContactObject2D) -> Array:
        return object_geom.signed_distance(self.points) - self.radii

    def active_contacts(self, object_geom: ContactObject2D, margin_tol: float = 0.25) -> "ActiveContacts":
        margins = self.margins(object_geom)
        active = margins <= float(margin_tol)
        points = self.points[active]
        if len(points) == 0:
            return ActiveContacts.empty()
        return ActiveContacts(
            points=object_geom.closest_point(points),
            normals=object_geom.normal(points),
            owner_ids=self.owner_ids[active],
            owner_names=tuple(np.asarray(self.owner_names, dtype=object)[active]),
            margins=margins[active],
            source_points=points,
            source_radii=self.radii[active],
        )


@dataclass(frozen=True)
class ActiveContacts:
    """Active contact samples with normals pointing outward from the object."""

    points: Array
    normals: Array
    owner_ids: Array
    owner_names: tuple[str, ...]
    margins: Array
    source_points: Array
    source_radii: Array

    @classmethod
    def empty(cls) -> "ActiveContacts":
        return cls(
            points=np.empty((0, 2), dtype=float),
            normals=np.empty((0, 2), dtype=float),
            owner_ids=np.empty(0, dtype=int),
            owner_names=(),
            margins=np.empty(0, dtype=float),
            source_points=np.empty((0, 2), dtype=float),
            source_radii=np.empty(0, dtype=float),
        )

    @property
    def max_penetration(self) -> float:
        if len(self.margins) == 0:
            return 0.0
        return float(np.max(np.maximum(0.0, -self.margins)))

    def representatives_by_owner(self) -> "ActiveContacts":
        """Keep the deepest/closest sample for each owner id/name pair."""
        if len(self.owner_ids) == 0:
            return ActiveContacts.empty()

        keep: list[int] = []
        keys = list(zip(self.owner_ids.tolist(), self.owner_names))
        for key in dict.fromkeys(keys):
            candidates = [idx for idx, candidate_key in enumerate(keys) if candidate_key == key]
            best = min(candidates, key=lambda idx: self.margins[idx])
            keep.append(best)
        keep_arr = np.asarray(keep, dtype=int)
        return ActiveContacts(
            points=self.points[keep_arr],
            normals=self.normals[keep_arr],
            owner_ids=self.owner_ids[keep_arr],
            owner_names=tuple(np.asarray(self.owner_names, dtype=object)[keep_arr]),
            margins=self.margins[keep_arr],
            source_points=self.source_points[keep_arr],
            source_radii=self.source_radii[keep_arr],
        )

    def append(self, other: "ActiveContacts") -> "ActiveContacts":
        if len(self.points) == 0:
            return other
        if len(other.points) == 0:
            return self
        return ActiveContacts(
            points=np.vstack((self.points, other.points)),
            normals=np.vstack((self.normals, other.normals)),
            owner_ids=np.concatenate((self.owner_ids, other.owner_ids)),
            owner_names=(*self.owner_names, *other.owner_names),
            margins=np.concatenate((self.margins, other.margins)),
            source_points=np.vstack((self.source_points, other.source_points)),
            source_radii=np.concatenate((self.source_radii, other.source_radii)),
        )


@dataclass(frozen=True)
class ContactProblem:
    """No-penetration problem generated from a model-specific sampler."""

    object_geom: ContactObject2D
    sample_fn: Callable[[Array], ContactSampleSet]
    active_margin_tol: float = 0.25

    def sample_set(self, q: Array) -> ContactSampleSet:
        return self.sample_fn(q)

    def margins(self, q: Array) -> Array:
        return self.sample_set(q).margins(self.object_geom)

    def active_contacts(self, q: Array) -> ActiveContacts:
        return self.sample_set(q).active_contacts(self.object_geom, self.active_margin_tol)


def sample_segment(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    owner_id: int,
    owner_name: str,
    samples: int = 25,
    endpoint_margin: float = 0.04,
) -> ContactSampleSet:
    """Sample a fixed capsule/segment contact primitive."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    start_arr = np.asarray(start, dtype=float).reshape(2)
    end_arr = np.asarray(end, dtype=float).reshape(2)
    fractions = np.linspace(float(endpoint_margin), 1.0 - float(endpoint_margin), int(samples))
    points = start_arr[None, :] + fractions[:, None] * (end_arr - start_arr)[None, :]
    return ContactSampleSet(
        points=points,
        radii=np.full(len(points), float(radius), dtype=float),
        owner_ids=np.full(len(points), int(owner_id), dtype=int),
        owner_names=tuple(str(owner_name) for _ in range(len(points))),
        fractions=fractions,
    )


def combine_sample_sets(sample_sets: Sequence[ContactSampleSet]) -> ContactSampleSet:
    """Combine independently sampled contact primitives."""
    non_empty = [sample for sample in sample_sets if len(sample.points)]
    if not non_empty:
        return ContactSampleSet(
            points=np.empty((0, 2), dtype=float),
            radii=np.empty(0, dtype=float),
            owner_ids=np.empty(0, dtype=int),
            owner_names=(),
            fractions=np.empty(0, dtype=float),
        )
    return ContactSampleSet(
        points=np.vstack([sample.points for sample in non_empty]),
        radii=np.concatenate([sample.radii for sample in non_empty]),
        owner_ids=np.concatenate([sample.owner_ids for sample in non_empty]),
        owner_names=tuple(name for sample in non_empty for name in sample.owner_names),
        fractions=np.concatenate([sample.fractions for sample in non_empty]),
    )
