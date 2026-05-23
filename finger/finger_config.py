"""Dataclass configuration layer for modular finger linkage CMA-ES search.

This module intentionally contains only data structures and deterministic
extended-pose helpers. It does not implement kinematic solving, animation,
plotting, objective functions, or the CMA-ES optimization loop.

Geometry convention for the fully extended reference pose:
    O = [0, 0]
    A = O + [OA, 0]
    B = A + [AB, 0]
    D = B + [BD, 0]

FingerSearchBounds defines where CMA-ES searches. Point clearance diameters
define feature sizes, and ClearanceRule defines which point pairs must be
checked. Diameter-based clearance and explicit minimum distance rules are both
converted to one required distance.

The clearance evaluator only requires a point_dict-like mapping. Some rules,
such as A vs B_prime, are constant with motion because B_prime is fixed
relative to A through AB_prime. Future rules between independently moving
bodies may need trajectory-wide evaluation once the kinematic model exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from typing import Iterable

import numpy as np


Array = np.ndarray
Vec2Like = Iterable[float]
ClearanceMetric = float | str

DESIGN_KEYS = ("OO_prime", "AA_prime", "AB_prime", "BC_prime")
DESIGN_VECTOR_SIZE = 2 * len(DESIGN_KEYS)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default_finger.yaml"


def as_vec2(v: Vec2Like, name: str = "vector") -> Array:
    arr = np.asarray(tuple(v), dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must have size 2, got {arr.size}")
    return arr


def as_tuple2(v: Vec2Like, name: str = "vector") -> tuple[float, float]:
    arr = as_vec2(v, name)
    return (float(arr[0]), float(arr[1]))


def checked_design_vector(x: Array, name: str = "design vector") -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != DESIGN_VECTOR_SIZE:
        raise ValueError(f"{name} must have size {DESIGN_VECTOR_SIZE}, got {arr.size}")
    return arr


@dataclass(frozen=True)
class FingerFixedConfig:
    """Fixed phalanx lengths in the fully extended reference pose."""

    OA: float = 40.0
    AB: float = 25.0
    BD: float = 25.0

    def __post_init__(self) -> None:
        for name in ("OA", "AB", "BD"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def O(self) -> Array:
        return np.array([0.0, 0.0], dtype=float)

    @property
    def A(self) -> Array:
        return np.array([self.OA, 0.0], dtype=float)

    @property
    def B(self) -> Array:
        return np.array([self.OA + self.AB, 0.0], dtype=float)

    @property
    def D(self) -> Array:
        return np.array([self.OA + self.AB + self.BD, 0.0], dtype=float)

    @property
    def OA_vec(self) -> Array:
        return np.array([self.OA, 0.0], dtype=float)

    @property
    def AB_vec(self) -> Array:
        return np.array([self.AB, 0.0], dtype=float)

    @property
    def BD_vec(self) -> Array:
        return np.array([self.BD, 0.0], dtype=float)


@dataclass(frozen=True)
class FingerDesignVariables:
    """Four 2D offsets that form the 8D optimizer vector."""

    OO_prime: tuple[float, float] = (10.0, 15.0)
    AA_prime: tuple[float, float] = (22.0, 8.0)
    AB_prime: tuple[float, float] = (5.0, -10.0)
    BC_prime: tuple[float, float] = (5.0, -8.0)

    def __post_init__(self) -> None:
        for key in DESIGN_KEYS:
            object.__setattr__(self, key, as_tuple2(getattr(self, key), key))

    def vec(self, key: str) -> Array:
        if key not in DESIGN_KEYS:
            raise KeyError(f"Unknown design variable: {key}")
        return as_vec2(getattr(self, key), key)

    @classmethod
    def from_vector(cls, x: Array) -> "FingerDesignVariables":
        x = checked_design_vector(x)
        return cls(**{key: tuple(x[2 * i : 2 * i + 2]) for i, key in enumerate(DESIGN_KEYS)})

    def to_vector(self) -> Array:
        return np.concatenate([self.vec(key) for key in DESIGN_KEYS])

    @property
    def OO_prime_vec(self) -> Array:
        return self.vec("OO_prime")

    @property
    def AA_prime_vec(self) -> Array:
        return self.vec("AA_prime")

    @property
    def AB_prime_vec(self) -> Array:
        return self.vec("AB_prime")

    @property
    def BC_prime_vec(self) -> Array:
        return self.vec("BC_prime")


@dataclass(frozen=True)
class Vector2SearchBounds:
    """2D rough search box for one design variable."""

    name: str
    lower: tuple[float, float]
    upper: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("search bound name must be nonempty")
        lower = as_tuple2(self.lower, f"{self.name}.lower")
        upper = as_tuple2(self.upper, f"{self.name}.upper")
        if np.any(as_vec2(lower) >= as_vec2(upper)):
            raise ValueError(f"{self.name}: lower must be elementwise smaller than upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def lower_vec(self) -> Array:
        return as_vec2(self.lower, f"{self.name}.lower")

    def upper_vec(self) -> Array:
        return as_vec2(self.upper, f"{self.name}.upper")

    def center_vec(self) -> Array:
        return 0.5 * (self.lower_vec() + self.upper_vec())

    def scale_vec(self) -> Array:
        return self.upper_vec() - self.lower_vec()


@dataclass(frozen=True)
class FingerSearchBounds:
    """
    Rough optimizer search box for CMA-ES.

    These bounds are not physical clearance constraints. They only tell CMA-ES
    where it is allowed to search; ClearanceRule handles non-interference.
    """

    OO_prime: Vector2SearchBounds = field(
        default_factory=lambda: Vector2SearchBounds("OO_prime", (-20.0, -35.0), (45.0, 45.0))
    )
    AA_prime: Vector2SearchBounds = field(
        default_factory=lambda: Vector2SearchBounds("AA_prime", (-5.0, -25.0), (40.0, 25.0))
    )
    AB_prime: Vector2SearchBounds = field(
        default_factory=lambda: Vector2SearchBounds("AB_prime", (-15.0, -35.0), (35.0, 35.0))
    )
    BC_prime: Vector2SearchBounds = field(
        default_factory=lambda: Vector2SearchBounds("BC_prime", (-15.0, -35.0), (35.0, 35.0))
    )
    normalized_sigma0: float = 0.3

    def __post_init__(self) -> None:
        for key in DESIGN_KEYS:
            bound = getattr(self, key)
            if bound.name != key:
                raise ValueError(f"{key} search bounds must have name {key}, got {bound.name}")
        if self.normalized_sigma0 <= 0.0:
            raise ValueError(f"normalized_sigma0 must be positive, got {self.normalized_sigma0}")

    def variable_names(self) -> tuple[str, ...]:
        return DESIGN_KEYS

    def lower_vector(self) -> Array:
        return np.concatenate([getattr(self, key).lower_vec() for key in DESIGN_KEYS])

    def upper_vector(self) -> Array:
        return np.concatenate([getattr(self, key).upper_vec() for key in DESIGN_KEYS])

    def center_vector(self) -> Array:
        return 0.5 * (self.lower_vector() + self.upper_vector())

    def scale_vector(self) -> Array:
        return self.upper_vector() - self.lower_vector()

    def contains(self, x: Array) -> bool:
        x = np.asarray(x, dtype=float).reshape(-1)
        return x.size == DESIGN_VECTOR_SIZE and bool(
            np.all(x >= self.lower_vector()) and np.all(x <= self.upper_vector())
        )

    def clip(self, x: Array) -> Array:
        x = checked_design_vector(x)
        return np.clip(x, self.lower_vector(), self.upper_vector())

    def decode(self, x: Array) -> FingerDesignVariables:
        return FingerDesignVariables.from_vector(x)

    def encode(self, design: FingerDesignVariables) -> Array:
        return checked_design_vector(design.to_vector())

    def normalize(self, x: Array) -> Array:
        x = checked_design_vector(x)
        return 2.0 * (x - self.lower_vector()) / self.scale_vector() - 1.0

    def denormalize(self, z: Array) -> Array:
        z = checked_design_vector(z, "normalized design vector")
        return self.lower_vector() + 0.5 * (z + 1.0) * self.scale_vector()

    def denormalize_clipped(self, z: Array) -> Array:
        """Denormalize and clip to bounds for debug / visualization only."""
        return self.clip(self.denormalize(z))

    def normalized_center(self) -> Array:
        return np.zeros(DESIGN_VECTOR_SIZE, dtype=float)

    def default_sigma0(self, normalized: bool = True) -> float:
        if normalized:
            return self.normalized_sigma0
        return float(np.mean(self.scale_vector()) * self.normalized_sigma0 / 2.0)


# Backward-compatible name for older code that used FingerDesignBounds.
FingerDesignBounds = FingerSearchBounds


@dataclass(frozen=True)
class ClearanceRule:
    """
    Minimum radial clearance between two named points.

    The current default rules are static/local clearances whose distances are
    fixed by the design vector. Future inter-body rules should use
    scope="trajectory" and be evaluated over solved kinematic poses.
    """

    point_a: str
    point_b: str
    name: str | None = None
    gap: float = 0.0
    min_distance: float | None = None
    scope: str = "static"

    def __post_init__(self) -> None:
        if not self.point_a:
            raise ValueError("point_a must be nonempty")
        if not self.point_b:
            raise ValueError("point_b must be nonempty")
        if self.scope not in ("static", "trajectory"):
            raise ValueError(f"scope must be 'static' or 'trajectory', got {self.scope}")
        if self.gap < 0.0:
            raise ValueError(f"gap must be nonnegative, got {self.gap}")
        if self.min_distance is not None and self.min_distance <= 0.0:
            raise ValueError(f"min_distance must be positive, got {self.min_distance}")

    @property
    def rule_name(self) -> str:
        return self.name or f"{self.point_a}__{self.point_b}"

    @property
    def mode(self) -> str:
        return "explicit_min_distance" if self.min_distance is not None else "diameter_based"


@dataclass(frozen=True)
class PointClearanceDiameters:
    """Clearance feature diameters keyed by point name."""

    values: dict[str, float] = field(
        default_factory=lambda: {
            "O": 6.0,
            "O_prime": 6.0,
            "A": 6.0,
            "A_prime": 6.0,
            "B": 6.0,
            "B_prime": 6.0,
            "C_prime": 6.0,
            "D": 6.0,
        }
    )

    def __post_init__(self) -> None:
        checked: dict[str, float] = {}
        for point, diameter in self.values.items():
            if not point:
                raise ValueError("clearance diameter point name must be nonempty")
            diameter = float(diameter)
            if diameter <= 0.0:
                raise ValueError(f"clearance diameter for {point} must be positive, got {diameter}")
            checked[str(point)] = diameter
        object.__setattr__(self, "values", checked)

    def diameter(self, point: str) -> float:
        try:
            return self.values[point]
        except KeyError as exc:
            raise KeyError(f"Missing clearance diameter for point {point}") from exc


@dataclass(frozen=True)
class FingerClearanceConfig:
    """Geometric non-interference rules for named points."""

    rules: tuple[ClearanceRule, ...] = (
        ClearanceRule(
            name="A_to_B_prime_clearance",
            point_a="A",
            point_b="B_prime",
            gap=1.0,
        ),
        ClearanceRule(
            name="A_prime_to_C_prime_min_distance",
            point_a="A_prime",
            point_b="C_prime",
            min_distance=8.0,
        ),
    )
    diameters: PointClearanceDiameters = field(default_factory=PointClearanceDiameters)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        for rule in self.rules:
            if rule.min_distance is None:
                self.diameters.diameter(rule.point_a)
                self.diameters.diameter(rule.point_b)


def clearance_required_distance(rule: ClearanceRule, diameters: PointClearanceDiameters) -> float:
    if rule.min_distance is not None:
        return float(rule.min_distance)
    return 0.5 * (diameters.diameter(rule.point_a) + diameters.diameter(rule.point_b)) + rule.gap


def clearance_distance(rule: ClearanceRule, points: dict[str, Array]) -> float:
    missing = [point for point in (rule.point_a, rule.point_b) if point not in points]
    if missing:
        raise KeyError(f"Clearance rule {rule.rule_name} references missing point(s): {missing}")
    return float(np.linalg.norm(points[rule.point_b] - points[rule.point_a]))


def clearance_margin(
    rule: ClearanceRule,
    points: dict[str, Array],
    diameters: PointClearanceDiameters | None = None,
) -> float:
    if rule.min_distance is None and diameters is None:
        raise ValueError("diameters are required for diameter-based clearance rules")
    if diameters is None:
        return clearance_distance(rule, points) - float(rule.min_distance)
    return clearance_distance(rule, points) - clearance_required_distance(rule, diameters)


def clearance_violation(
    rule: ClearanceRule,
    points: dict[str, Array],
    diameters: PointClearanceDiameters | None = None,
) -> float:
    return max(0.0, -clearance_margin(rule, points, diameters))


def evaluate_clearances(
    clearance_config: FingerClearanceConfig,
    points: dict[str, Array],
) -> dict[str, dict[str, ClearanceMetric]]:
    metrics: dict[str, dict[str, ClearanceMetric]] = {}
    for rule in clearance_config.rules:
        distance = clearance_distance(rule, points)
        required_distance = clearance_required_distance(rule, clearance_config.diameters)
        margin = distance - required_distance
        metrics[rule.rule_name] = {
            "point_a": rule.point_a,
            "point_b": rule.point_b,
            "mode": rule.mode,
            "scope": rule.scope,
            "distance": distance,
            "required_distance": required_distance,
            "margin": margin,
            "violation": max(0.0, -margin),
        }
    return metrics


@dataclass(frozen=True)
class FingerExtendedGeometry:
    """Deterministic extended-pose points and linkage lengths."""

    fixed: FingerFixedConfig = field(default_factory=FingerFixedConfig)
    design: FingerDesignVariables = field(default_factory=FingerDesignVariables)

    @property
    def O(self) -> Array:
        return self.fixed.O

    @property
    def A(self) -> Array:
        return self.fixed.A

    @property
    def B(self) -> Array:
        return self.fixed.B

    @property
    def D(self) -> Array:
        return self.fixed.D

    @property
    def O_prime(self) -> Array:
        return self.O + self.design.OO_prime_vec

    @property
    def A_prime(self) -> Array:
        return self.A + self.design.AA_prime_vec

    @property
    def B_prime(self) -> Array:
        return self.A + self.design.AB_prime_vec

    @property
    def C_prime(self) -> Array:
        return self.B + self.design.BC_prime_vec

    @property
    def O_prime_B_prime_length(self) -> float:
        return float(np.linalg.norm(self.B_prime - self.O_prime))

    @property
    def A_prime_C_prime_length(self) -> float:
        return float(np.linalg.norm(self.C_prime - self.A_prime))

    @property
    def AB_prime_length(self) -> float:
        return float(np.linalg.norm(self.design.AB_prime_vec))

    @property
    def BC_prime_length(self) -> float:
        return float(np.linalg.norm(self.design.BC_prime_vec))

    def point_dict(self) -> dict[str, Array]:
        return {
            "O": self.O,
            "A": self.A,
            "B": self.B,
            "D": self.D,
            "O_prime": self.O_prime,
            "A_prime": self.A_prime,
            "B_prime": self.B_prime,
            "C_prime": self.C_prime,
        }

    def length_dict(self) -> dict[str, float]:
        return {
            "OA": self.fixed.OA,
            "AB": self.fixed.AB,
            "BD": self.fixed.BD,
            "AB_prime": self.AB_prime_length,
            "BC_prime": self.BC_prime_length,
            "O_prime_B_prime": self.O_prime_B_prime_length,
            "A_prime_C_prime": self.A_prime_C_prime_length,
        }

    def clearance_metrics(
        self,
        clearance_config: FingerClearanceConfig,
    ) -> dict[str, dict[str, ClearanceMetric]]:
        return evaluate_clearances(clearance_config, self.point_dict())

    def is_clearance_valid(
        self,
        clearance_config: FingerClearanceConfig,
        tol: float = 0.0,
    ) -> bool:
        metrics = self.clearance_metrics(clearance_config)
        return all(float(rule_metrics["margin"]) >= -tol for rule_metrics in metrics.values())


@dataclass(frozen=True)
class FingerCandidateCheck:
    """Lightweight candidate validation result for later optimizer code."""

    design: FingerDesignVariables
    geometry: FingerExtendedGeometry
    in_search_bounds: bool
    clearance_valid: bool
    clearance_metrics: dict[str, dict[str, ClearanceMetric]]

    @property
    def valid(self) -> bool:
        return self.in_search_bounds and self.clearance_valid


@dataclass(frozen=True)
class FingerConfig:
    """Top-level configuration loaded from YAML or assembled in Python."""

    fixed: FingerFixedConfig = field(default_factory=FingerFixedConfig)
    initial_design: FingerDesignVariables = field(default_factory=FingerDesignVariables)
    search_bounds: FingerSearchBounds = field(default_factory=FingerSearchBounds)
    clearance: FingerClearanceConfig = field(default_factory=FingerClearanceConfig)


def check_candidate(
    x: Array,
    fixed: FingerFixedConfig,
    search_bounds: FingerSearchBounds,
    clearance_config: FingerClearanceConfig,
) -> FingerCandidateCheck:
    x = checked_design_vector(x)
    design = search_bounds.decode(x)
    geometry = FingerExtendedGeometry(fixed=fixed, design=design)
    clearance_metrics = geometry.clearance_metrics(clearance_config)
    clearance_valid = all(float(metrics["margin"]) >= 0.0 for metrics in clearance_metrics.values())
    return FingerCandidateCheck(
        design=design,
        geometry=geometry,
        in_search_bounds=search_bounds.contains(x),
        clearance_valid=clearance_valid,
        clearance_metrics=clearance_metrics,
    )


def _require_mapping(data: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return data


def fixed_config_from_mapping(data: Mapping[str, Any]) -> FingerFixedConfig:
    return FingerFixedConfig(
        OA=float(data["OA"]),
        AB=float(data["AB"]),
        BD=float(data["BD"]),
    )


def design_variables_from_mapping(data: Mapping[str, Any]) -> FingerDesignVariables:
    return FingerDesignVariables(**{key: as_tuple2(data[key], key) for key in DESIGN_KEYS})


def search_bounds_from_mapping(data: Mapping[str, Any]) -> FingerSearchBounds:
    kwargs: dict[str, Any] = {}
    for key in DESIGN_KEYS:
        bound_data = _require_mapping(data[key], f"search_bounds.{key}")
        kwargs[key] = Vector2SearchBounds(
            name=key,
            lower=as_tuple2(bound_data["lower"], f"search_bounds.{key}.lower"),
            upper=as_tuple2(bound_data["upper"], f"search_bounds.{key}.upper"),
        )
    kwargs["normalized_sigma0"] = float(data.get("normalized_sigma0", 0.3))
    return FingerSearchBounds(**kwargs)


def clearance_rule_from_mapping(data: Mapping[str, Any]) -> ClearanceRule:
    if "points" in data:
        pair = tuple(data["points"])
        if len(pair) != 2:
            raise ValueError("clearance rule points must contain exactly two point names")
        point_a, point_b = pair
    else:
        point_a = data["point_a"]
        point_b = data["point_b"]
    return ClearanceRule(
        name=data.get("name"),
        point_a=str(point_a),
        point_b=str(point_b),
        gap=float(data.get("gap", 0.0)),
        min_distance=None if data.get("min_distance") is None else float(data["min_distance"]),
        scope=str(data.get("scope", "static")),
    )


def point_clearance_diameters_from_mapping(data: Mapping[str, Any]) -> PointClearanceDiameters:
    return PointClearanceDiameters(values={str(point): float(diameter) for point, diameter in data.items()})


def clearance_config_from_mapping(data: Any, diameters_data: Any | None = None) -> FingerClearanceConfig:
    if not isinstance(data, list | tuple):
        raise ValueError("clearance_rules must be a list")
    diameters = (
        PointClearanceDiameters()
        if diameters_data is None
        else point_clearance_diameters_from_mapping(
            _require_mapping(diameters_data, "clearance_diameters")
        )
    )
    return FingerClearanceConfig(
        diameters=diameters,
        rules=tuple(clearance_rule_from_mapping(_require_mapping(rule, "clearance rule")) for rule in data)
    )


def finger_config_from_mapping(data: Mapping[str, Any]) -> FingerConfig:
    return FingerConfig(
        fixed=fixed_config_from_mapping(_require_mapping(data["fixed"], "fixed")),
        initial_design=design_variables_from_mapping(
            _require_mapping(data["initial_design"], "initial_design")
        ),
        search_bounds=search_bounds_from_mapping(_require_mapping(data["search_bounds"], "search_bounds")),
        clearance=clearance_config_from_mapping(
            data["clearance_rules"],
            data.get("clearance_diameters"),
        ),
    )


def load_finger_config(path: str | Path = DEFAULT_CONFIG_PATH) -> FingerConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("load_finger_config requires PyYAML. Install package 'pyyaml'.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return finger_config_from_mapping(_require_mapping(data, str(config_path)))


def example() -> None:
    config = load_finger_config()
    fixed = config.fixed
    search_bounds = config.search_bounds
    clearance_config = config.clearance
    x = config.initial_design.to_vector()

    design = search_bounds.decode(x)
    geometry = FingerExtendedGeometry(fixed=fixed, design=design)
    candidate = check_candidate(x, fixed, search_bounds, clearance_config)

    print("Points:")
    for name, point in geometry.point_dict().items():
        print(f"  {name:8s}: {point}")

    print("\nLengths:")
    for name, length in geometry.length_dict().items():
        print(f"  {name:24s}: {length:.4f}")

    print("\nSearch bounds:")
    print("  lower:", search_bounds.lower_vector())
    print("  upper:", search_bounds.upper_vector())
    print("  inside:", candidate.in_search_bounds)

    print("\nClearances:")
    for name, metrics in candidate.clearance_metrics.items():
        print(f"  {name}:")
        print(f"    points:            {metrics['point_a']} -> {metrics['point_b']}")
        print(f"    mode:              {metrics['mode']}")
        print(f"    scope:             {metrics['scope']}")
        print(f"    distance:          {float(metrics['distance']):.4f}")
        print(f"    required_distance: {float(metrics['required_distance']):.4f}")
        print(f"    margin:            {float(metrics['margin']):.4f}")
        print(f"    violation:         {float(metrics['violation']):.4f}")
    print("clearance valid:", candidate.clearance_valid)


if __name__ == "__main__":
    example()
