#!/usr/bin/env python3
"""
stl2scad_parametric.py

Heuristic STL to OpenSCAD converter.

This tool attempts a lightweight "parametric rebuild" of simple STL meshes into
editable OpenSCAD primitives. If a clean primitive cannot be recognized, it can
fall back to an embedded OpenSCAD polyhedron().

Supported reconstruction heuristics:
  - axis-aligned boxes
  - vertical cylinders
  - rotated cylinders using PCA
  - axis-aligned box-with-through-hole shapes as difference()
  - simple cylindrical shells as difference()
  - spheres
  - extruded 2D profiles, when the mesh has clear top/bottom planar faces
  - aggressive 2D silhouette extrusion fallback
  - mesh fallback as polyhedron()

Install:
  python3 -m pip install numpy trimesh scipy

Usage:
  python3 stl2scad_parametric.py input.stl -o output.scad
  python3 stl2scad_parametric.py input.stl -o output.scad --strategy primitive
  python3 stl2scad_parametric.py input.stl -o output.scad --strategy polyhedron
  python3 stl2scad_parametric.py input.stl -o output.scad --tolerance 0.03
  python3 stl2scad_parametric.py input.stl -o output.scad --aggressive --diagnose

Important:
  STL files contain triangles, not feature history. True parametric CAD
  reconstruction is not generally solvable from STL alone. This script is a
  pragmatic heuristic converter for simple mechanical models.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency: trimesh. Install with:\n"
        "  python3 -m pip install numpy trimesh scipy"
    ) from exc


@dataclass
class Detection:
    kind: str
    confidence: float
    scad: str
    notes: list[str]


@dataclass
class Hole:
    axis: int
    center: np.ndarray
    radius: float
    height: float
    confidence: float


def fmt(value: float, precision: int = 6) -> str:
    """Format a number for compact SCAD output."""
    if abs(value) < 10 ** (-precision):
        value = 0.0
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def vec(values: Iterable[float]) -> str:
    return "[" + ", ".join(fmt(float(v)) for v in values) + "]"


def matrix4(values: np.ndarray) -> str:
    rows = []
    for row in values:
        rows.append("[" + ", ".join(fmt(float(value)) for value in row) + "]")
    return "[" + ", ".join(rows) + "]"


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def oriented_cylinder_scad(
    center: np.ndarray,
    axis: np.ndarray,
    height: float,
    radius: float,
    fragments: int,
    indent: str = "",
) -> str:
    """Return SCAD for a cylinder whose local Z axis is rotated onto axis."""
    axis = unit(np.asarray(axis, dtype=float))
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    rot_axis = np.cross(z_axis, axis)

    cylinder = f"cylinder(h={fmt(height)}, r={fmt(radius)}, center=true, $fn={fragments});"
    if np.linalg.norm(rot_axis) < 1e-9:
        if dot < 0:
            body = f"rotate(a=180, v=[1, 0, 0]) {cylinder}"
        else:
            body = cylinder
    else:
        body = f"rotate(a={fmt(angle)}, v={vec(rot_axis)}) {cylinder}"
    return f"{indent}translate({vec(center)}) {body}\n"


def axis_aligned_cylinder_scad(
    axis: int,
    center: np.ndarray,
    height: float,
    radius: float,
    fragments: int,
    indent: str = "",
) -> str:
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    return oriented_cylinder_scad(center, axes[axis], height, radius, fragments, indent)


def fit_circle_2d(points: np.ndarray) -> Optional[tuple[np.ndarray, float, float]]:
    """Least-squares 2D circle fit: returns center, radius, relative error."""
    if len(points) < 6:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = cx * cx + cy * cy + c
    if radius_sq <= 0:
        return None
    center = np.array([cx, cy])
    radius = float(math.sqrt(radius_sq))
    distances = np.linalg.norm(points - center, axis=1)
    error = float(np.median(np.abs(distances - radius)) / max(radius, 1e-9))
    return center, radius, error


def silhouette_polygon_points(points: np.ndarray, tolerance: float, max_points: int = 220) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    mins = points.min(axis=0)
    spans = np.ptp(points, axis=0)
    if np.any(spans <= tolerance):
        return None

    resolution = 512
    scale = (resolution - 20) / float(max(spans))
    image_points = np.round((points - mins) * scale + 10).astype(np.int32)
    image = np.zeros((resolution, resolution), dtype=np.uint8)
    radius = max(2, int(scale * max(tolerance, max(spans) * 0.008)))
    for point in image_points:
        cv2.circle(image, tuple(point), radius, 255, -1)

    kernel_size = max(3, int(scale * max(tolerance * 2.0, max(spans) * 0.015)))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.dilate(image, kernel, iterations=1)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) <= 0:
        return None

    epsilon = max(1.0, 0.0075 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    model_points = (approx.astype(float) - 10.0) / scale + mins
    if len(model_points) > max_points:
        step = int(math.ceil(len(model_points) / max_points))
        model_points = model_points[::step]
    return model_points


def silhouette_multmatrix(axis: int, start: float) -> np.ndarray:
    if axis == 0:
        return np.array(
            [
                [0.0, 0.0, 1.0, start],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    if axis == 1:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, start],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, start],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def silhouette_extrude_scad(axis: int, bounds: np.ndarray, extents: np.ndarray, model_points: np.ndarray, indent: str = "") -> str:
    points_text = ",\n".join(f"{indent}            {vec(point)}" for point in model_points)
    transform = silhouette_multmatrix(axis, float(bounds[0, axis]))
    return f"""{indent}multmatrix({matrix4(transform)})
{indent}    linear_extrude(height={fmt(float(extents[axis]))})
{indent}        polygon(points=[
{points_text}
{indent}        ]);
"""


def scad_header(source: Path, detection: Detection) -> str:
    notes = "\n".join(f"// - {line}" for line in detection.notes)
    return f"""// Generated by stl2scad_parametric.py
// Source: {source.name}
// Strategy: {detection.kind}
// Confidence: {detection.confidence:.2f}
{notes}

"""


def mesh_diagnostics(mesh: "trimesh.Trimesh", tolerance: float) -> list[str]:
    bounds = mesh.bounds
    extents = mesh.extents
    notes = [
        f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.",
        f"Bounds min={vec(bounds[0])}, max={vec(bounds[1])}, extents={vec(extents)}.",
        f"Watertight={bool(getattr(mesh, 'is_watertight', False))}, volume={fmt(float(getattr(mesh, 'volume', 0.0)))}.",
    ]

    if len(mesh.vertices) > 5000:
        notes.append("High triangle count can hide simple primitives; try simplifying/remeshing before conversion.")
    if not bool(getattr(mesh, "is_watertight", False)):
        notes.append("Mesh is not watertight; boolean and volume-based detection will be less reliable.")
    if np.any(extents <= tolerance):
        notes.append("One or more dimensions are near zero compared with tolerance.")
    notes.append("If this falls back to polyhedron(), retry with --aggressive and a larger --tolerance, for example --tolerance 0.05.")
    return notes


def load_mesh(path: Path) -> "trimesh.Trimesh":
    mesh = trimesh.load_mesh(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"No mesh data found in {path}")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded file did not produce a single triangular mesh")
    # Trimesh cleanup APIs have changed across releases. Use the stable
    # operations when present, but keep loading functional if a method is
    # unavailable in the installed version.
    for method_name in (
        "remove_duplicate_faces",
        "remove_degenerate_faces",
        "remove_unreferenced_vertices",
    ):
        method = getattr(mesh, method_name, None)
        if callable(method):
            method()
    return mesh


def axis_aligned_box(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    bounds = mesh.bounds
    extents = mesh.extents
    if np.any(extents <= 0):
        return None

    vertices = mesh.vertices
    min_corner, max_corner = bounds

    on_any_box_plane = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        on_any_box_plane |= np.isclose(vertices[:, axis], min_corner[axis], atol=tolerance)
        on_any_box_plane |= np.isclose(vertices[:, axis], max_corner[axis], atol=tolerance)

    box_plane_ratio = float(np.mean(on_any_box_plane))
    expected_faces = 12
    face_score = min(1.0, expected_faces / max(expected_faces, len(mesh.faces)))
    confidence = 0.65 * box_plane_ratio + 0.35 * face_score

    if confidence < 0.86:
        return None

    center = (min_corner + max_corner) / 2.0
    scad = f"translate({vec(center)}) cube({vec(extents)}, center=true);\n"
    return Detection(
        kind="axis_aligned_box",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected an axis-aligned rectangular solid.",
            "Output uses cube(size, center=true), translated to the STL bounds center.",
        ],
    )


def oriented_bounding_box(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    """Approximate a part with an oriented box when PCA/OBB fit is strong enough."""
    if len(mesh.vertices) < 8:
        return None
    try:
        box = mesh.bounding_box_oriented
        transform = np.array(box.primitive.transform, dtype=float)
        extents = np.array(box.primitive.extents, dtype=float)
    except Exception:
        return None

    if np.any(extents <= tolerance):
        return None

    box_volume = float(np.prod(extents))
    if box_volume <= tolerance:
        return None

    mesh_volume = abs(float(getattr(mesh, "volume", 0.0)))
    if mesh_volume > tolerance:
        fill_ratio = min(1.0, mesh_volume / box_volume)
    else:
        # Non-watertight meshes often report unusable volume. Fall back to a
        # surface heuristic: many box-like meshes have vertices near OBB planes.
        inv = np.linalg.inv(transform)
        homogeneous = np.column_stack((mesh.vertices, np.ones(len(mesh.vertices))))
        local = (inv @ homogeneous.T).T[:, :3]
        half = extents / 2.0
        near_plane = np.zeros(len(local), dtype=bool)
        for axis in range(3):
            near_plane |= np.isclose(np.abs(local[:, axis]), half[axis], atol=max(tolerance, extents[axis] * 0.025))
        fill_ratio = float(np.mean(near_plane))

    confidence = 0.55 + 0.42 * min(1.0, fill_ratio)
    if confidence < 0.72:
        return None

    scad = f"multmatrix({matrix4(transform)}) cube({vec(extents)}, center=true);\n"
    return Detection(
        kind="oriented_bounding_box_approximation",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected a strong oriented bounding-box fit.",
            "Output uses multmatrix(...) cube(...), which is an approximation but remains editable.",
            "Use this mainly with --aggressive for rotated boxes or blocky parts with bevels.",
        ],
    )


def vertical_cylinder(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    bounds = mesh.bounds
    height = bounds[1, 2] - bounds[0, 2]
    if height <= tolerance:
        return None

    xy = vertices[:, :2]
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    radius = float(np.median(radii))
    if radius <= tolerance:
        return None

    radial_error = float(np.median(np.abs(radii - radius)) / radius)
    top_bottom_ratio = float(
        np.mean(
            np.isclose(vertices[:, 2], bounds[0, 2], atol=tolerance)
            | np.isclose(vertices[:, 2], bounds[1, 2], atol=tolerance)
        )
    )
    circularity_score = max(0.0, 1.0 - radial_error * 10.0)
    confidence = 0.75 * circularity_score + 0.25 * top_bottom_ratio

    if confidence < 0.84:
        return None

    z_mid = float((bounds[0, 2] + bounds[1, 2]) / 2.0)
    fragments = max(32, min(256, int(len(np.unique(np.round(xy, 4), axis=0)) * 1.5)))
    scad = (
        f"translate([{fmt(center[0])}, {fmt(center[1])}, {fmt(z_mid)}]) "
        f"cylinder(h={fmt(height)}, r={fmt(radius)}, center=true, $fn={fragments});\n"
    )
    return Detection(
        kind="vertical_cylinder",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected a cylinder aligned to the Z axis.",
            "Output uses cylinder(h, r, center=true).",
        ],
    )


def pca_oriented_cylinder(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    if len(vertices) < 12:
        return None

    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    best: Optional[tuple[float, np.ndarray, float, float, int]] = None
    for candidate_axis in vh:
        axis = unit(candidate_axis)
        projection = centered @ axis
        height = float(projection.max() - projection.min())
        if height <= tolerance:
            continue

        radial_vectors = centered - np.outer(projection, axis)
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        outer_radius = float(np.percentile(radial_distances, 75))
        if outer_radius <= tolerance:
            continue

        side_mask = radial_distances > outer_radius * 0.55
        if np.mean(side_mask) < 0.45:
            continue

        side_radii = radial_distances[side_mask]
        radial_error = float(np.median(np.abs(side_radii - np.median(side_radii))) / outer_radius)
        cap_ratio = float(
            np.mean(
                np.isclose(projection, projection.min(), atol=max(tolerance, height * 0.015))
                | np.isclose(projection, projection.max(), atol=max(tolerance, height * 0.015))
            )
        )
        elongation_score = min(1.0, height / max(outer_radius * 2.0, 1e-9))
        circularity_score = max(0.0, 1.0 - radial_error * 8.0)
        confidence = 0.68 * circularity_score + 0.18 * cap_ratio + 0.14 * elongation_score

        if best is None or confidence > best[0]:
            fragments = max(32, min(256, int(len(np.unique(np.round(radial_vectors[side_mask], 4), axis=0)) * 1.2)))
            best = (confidence, axis, height, float(np.median(side_radii)), fragments)

    if best is None:
        return None

    confidence, axis, height, radius, fragments = best
    if confidence < 0.84:
        return None

    scad = oriented_cylinder_scad(centroid, axis, height, radius, fragments)
    return Detection(
        kind="pca_oriented_cylinder",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected a cylinder whose axis is not necessarily aligned to world Z.",
            "Axis orientation was estimated with PCA, then emitted as rotate(a, v) cylinder(...).",
        ],
    )


def sphere(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    center = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - center, axis=1)
    radius = float(np.median(radii))
    if radius <= tolerance:
        return None

    radial_error = float(np.median(np.abs(radii - radius)) / radius)
    extents = mesh.extents
    extent_uniformity = float(np.std(extents) / np.mean(extents))
    confidence = max(0.0, 1.0 - radial_error * 8.0 - extent_uniformity * 2.0)

    if confidence < 0.82:
        return None

    fragments = max(48, min(256, int(math.sqrt(len(mesh.faces)) * 6)))
    scad = f"translate({vec(center)}) sphere(r={fmt(radius)}, $fn={fragments});\n"
    return Detection(
        kind="sphere",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected an approximately spherical mesh.",
            "Output uses sphere(r), translated to the estimated center.",
        ],
    )


def extruded_profile(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    z_min, z_max = mesh.bounds[:, 2]
    height = float(z_max - z_min)
    if height <= tolerance:
        return None

    bottom_mask = np.isclose(vertices[:, 2], z_min, atol=tolerance)
    top_mask = np.isclose(vertices[:, 2], z_max, atol=tolerance)
    if np.mean(bottom_mask) < 0.15 or np.mean(top_mask) < 0.15:
        return None

    profile_points = vertices[bottom_mask | top_mask][:, :2]
    if len(profile_points) < 6:
        return None

    # Convex hull approximation is intentional: it produces editable SCAD for
    # simple extrusions but will not preserve concave cutouts.
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    unique = np.unique(np.round(profile_points, 6), axis=0)
    if len(unique) < 3:
        return None

    try:
        hull = ConvexHull(unique)
    except Exception:
        return None

    hull_points = unique[hull.vertices]
    area = float(hull.volume)  # In 2D scipy stores area in .volume.
    bbox_area = float(np.prod(np.ptp(unique, axis=0)))
    if bbox_area <= tolerance:
        return None

    fill_ratio = area / bbox_area
    face_normals = mesh.face_normals
    vertical_side_ratio = float(np.mean(np.abs(face_normals[:, 2]) < 0.25))
    confidence = min(0.95, 0.55 + 0.25 * vertical_side_ratio + 0.15 * min(fill_ratio, 1.0))

    if confidence < 0.75:
        return None

    points_text = ",\n        ".join(vec(point) for point in hull_points)
    z_translate = fmt(z_min)
    scad = f"""translate([0, 0, {z_translate}])
    linear_extrude(height={fmt(height)})
        polygon(points=[
        {points_text}
        ]);
"""
    return Detection(
        kind="convex_extruded_profile",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected a likely Z-axis extrusion and rebuilt the convex hull of its 2D footprint.",
            "Concave details, holes, bevels, and fillets are not preserved by this heuristic.",
        ],
    )


def silhouette_extrusion(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    """Approximate complex parts as a concave 2D projection extruded through thickness.

    This is intentionally a fallback-style detector. It does not preserve all
    height changes or internal details, but it often gives a far more editable
    starting point than a huge polyhedron for brackets, hooks, and clamp shapes.
    It requires OpenCV if used locally.
    """
    vertices = mesh.vertices
    bounds = mesh.bounds
    extents = mesh.extents
    if len(vertices) < 20 or np.any(extents <= tolerance):
        return None

    # Prefer projecting along the smallest dimension for flat/bracket-like
    # parts; otherwise use Z to preserve the most familiar XY footprint.
    axis = int(np.argmin(extents))
    perpendicular = [index for index in range(3) if index != axis]
    model_points = silhouette_polygon_points(vertices[:, perpendicular], tolerance)
    if model_points is None:
        return None

    axis_name = "XYZ"[axis]
    scad = silhouette_extrude_scad(axis, bounds, extents, model_points)
    fill_ratio = len(model_points) / 220.0
    confidence = min(0.70, 0.45 + fill_ratio)
    return Detection(
        kind="silhouette_extrusion_approximation",
        confidence=confidence,
        scad=scad,
        notes=[
            f"Built an aggressive silhouette approximation by projecting the mesh and extruding along {axis_name}.",
            "This is not a true feature-history rebuild, but it is editable OpenSCAD and avoids giant polyhedron() output.",
            "Internal holes, ribs, bosses, stepped heights, and screw details may need to be remodeled manually.",
        ],
    )


def visual_hull_silhouette(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    bounds = mesh.bounds
    extents = mesh.extents
    if len(vertices) < 50 or np.any(extents <= tolerance):
        return None

    scad_parts: list[str] = []
    point_counts: list[int] = []
    for axis in range(3):
        perpendicular = [index for index in range(3) if index != axis]
        model_points = silhouette_polygon_points(vertices[:, perpendicular], tolerance, max_points=180)
        if model_points is None:
            return None
        point_counts.append(len(model_points))
        scad_parts.append(silhouette_extrude_scad(axis, bounds, extents, model_points, indent="    "))

    scad = "intersection() {\n" + "".join(scad_parts) + "}\n"
    confidence = min(0.74, 0.58 + min(point_counts) / 500.0)
    return Detection(
        kind="visual_hull_silhouette_approximation",
        confidence=confidence,
        scad=scad,
        notes=[
            "Built an aggressive visual-hull approximation from X, Y, and Z projected silhouettes.",
            "This usually matches complex bracket/clamp outlines better than a single extrusion and avoids giant polyhedron() output.",
            "It remains an approximation: fine ribs, screw details, holes, fillets, and stepped surfaces may need manual OpenSCAD cleanup.",
        ],
    )


def connected_face_components(mesh: "trimesh.Trimesh", face_indices: np.ndarray) -> list[np.ndarray]:
    selected = set(int(index) for index in face_indices)
    if not selected:
        return []

    adjacency: dict[int, list[int]] = {index: [] for index in selected}
    for first, second in getattr(mesh, "face_adjacency", []):
        first = int(first)
        second = int(second)
        if first in selected and second in selected:
            adjacency[first].append(second)
            adjacency[second].append(first)

    components: list[np.ndarray] = []
    seen: set[int] = set()
    for start in selected:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.array(component, dtype=int))
    return components


def detect_axis_aligned_through_holes(mesh: "trimesh.Trimesh", tolerance: float) -> list[Hole]:
    """Detect simple cylindrical through-holes parallel to X, Y, or Z."""
    vertices = mesh.vertices
    bounds = mesh.bounds
    centers = mesh.triangles_center
    normals = mesh.face_normals
    holes: list[Hole] = []

    for axis in range(3):
        perpendicular = [index for index in range(3) if index != axis]
        length = float(bounds[1, axis] - bounds[0, axis])
        if length <= tolerance:
            continue

        side_like = np.abs(normals[:, axis]) < 0.30
        interior = np.ones(len(mesh.faces), dtype=bool)
        for other_axis in perpendicular:
            span = float(bounds[1, other_axis] - bounds[0, other_axis])
            margin = max(tolerance * 3.0, span * 0.03)
            interior &= centers[:, other_axis] > bounds[0, other_axis] + margin
            interior &= centers[:, other_axis] < bounds[1, other_axis] - margin

        candidate_faces = np.where(side_like & interior)[0]
        for component in connected_face_components(mesh, candidate_faces):
            if len(component) < 8:
                continue
            component_vertices = np.unique(mesh.faces[component].reshape(-1))
            points = vertices[component_vertices]
            axis_span = float(points[:, axis].max() - points[:, axis].min())
            if axis_span < length * 0.65:
                continue

            fit = fit_circle_2d(points[:, perpendicular])
            if fit is None:
                continue
            center_2d, radius, error = fit
            if radius <= tolerance or error > 0.08:
                continue

            full_center = np.zeros(3)
            full_center[axis] = float((bounds[0, axis] + bounds[1, axis]) / 2.0)
            full_center[perpendicular[0]] = center_2d[0]
            full_center[perpendicular[1]] = center_2d[1]

            confidence = max(0.0, 1.0 - error * 8.0) * min(1.0, axis_span / length)
            holes.append(
                Hole(
                    axis=axis,
                    center=full_center,
                    radius=float(radius),
                    height=length + max(tolerance * 4.0, length * 0.05),
                    confidence=confidence,
                )
            )

    # Deduplicate the same hole found through nearby connected components.
    deduped: list[Hole] = []
    for hole in sorted(holes, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in deduped:
            if hole.axis != existing.axis:
                continue
            same_center = np.linalg.norm(hole.center - existing.center) < max(tolerance * 5.0, hole.radius * 0.15)
            same_radius = abs(hole.radius - existing.radius) < max(tolerance * 3.0, hole.radius * 0.10)
            if same_center and same_radius:
                duplicate = True
                break
        if not duplicate:
            deduped.append(hole)
    return deduped


def box_with_holes(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    holes = [hole for hole in detect_axis_aligned_through_holes(mesh, tolerance) if hole.confidence >= 0.62]
    if not holes:
        return None

    bounds = mesh.bounds
    extents = mesh.extents
    vertices = mesh.vertices
    if np.any(extents <= tolerance):
        return None

    on_box_plane = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        on_box_plane |= np.isclose(vertices[:, axis], bounds[0, axis], atol=max(tolerance, extents[axis] * 0.01))
        on_box_plane |= np.isclose(vertices[:, axis], bounds[1, axis], atol=max(tolerance, extents[axis] * 0.01))
    box_plane_ratio = float(np.mean(on_box_plane))
    if box_plane_ratio < 0.55:
        return None

    center = (bounds[0] + bounds[1]) / 2.0
    fragments = 96
    subtractors = "".join(
        axis_aligned_cylinder_scad(hole.axis, hole.center, hole.height, hole.radius, fragments, indent="    ")
        for hole in holes
    )
    scad = f"""difference() {{
    translate({vec(center)}) cube({vec(extents)}, center=true);
{subtractors}}}
"""
    confidence = min(0.97, 0.55 + 0.25 * box_plane_ratio + 0.20 * max(hole.confidence for hole in holes))
    return Detection(
        kind="box_with_cylindrical_holes_difference",
        confidence=confidence,
        scad=scad,
        notes=[
            f"Detected an axis-aligned box with {len(holes)} cylindrical through-hole(s).",
            "Output uses difference() with cube() minus cylinder() subtractors.",
            "Hole detection is conservative and targets clean through-holes parallel to X, Y, or Z.",
        ],
    )


def cylindrical_shell_difference(mesh: "trimesh.Trimesh", tolerance: float) -> Optional[Detection]:
    vertices = mesh.vertices
    if len(vertices) < 24:
        return None

    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    best: Optional[tuple[float, np.ndarray, float, float, float, int]] = None
    for candidate_axis in vh:
        axis = unit(candidate_axis)
        projection = centered @ axis
        height = float(projection.max() - projection.min())
        if height <= tolerance:
            continue
        radial_vectors = centered - np.outer(projection, axis)
        radii = np.linalg.norm(radial_vectors, axis=1)
        outer_radius = float(np.percentile(radii, 90))
        inner_candidates = radii[(radii > outer_radius * 0.12) & (radii < outer_radius * 0.82)]
        if outer_radius <= tolerance or len(inner_candidates) < max(8, len(radii) * 0.12):
            continue
        inner_radius = float(np.median(inner_candidates))
        if inner_radius <= tolerance or inner_radius >= outer_radius * 0.85:
            continue

        outer_mask = radii > outer_radius * 0.82
        inner_mask = np.abs(radii - inner_radius) < max(tolerance * 4.0, inner_radius * 0.12)
        outer_error = float(np.median(np.abs(radii[outer_mask] - np.median(radii[outer_mask]))) / outer_radius)
        inner_error = float(np.median(np.abs(radii[inner_mask] - inner_radius)) / inner_radius) if np.any(inner_mask) else 1.0
        confidence = max(0.0, 1.0 - outer_error * 7.0 - inner_error * 4.0)
        if best is None or confidence > best[0]:
            fragments = max(48, min(256, int(len(np.unique(np.round(radial_vectors[outer_mask], 4), axis=0)) * 1.1)))
            best = (confidence, axis, height, outer_radius, inner_radius, fragments)

    if best is None:
        return None

    confidence, axis, height, outer_radius, inner_radius, fragments = best
    if confidence < 0.78:
        return None

    subtract_height = height + max(tolerance * 4.0, height * 0.05)
    scad = "difference() {\n"
    scad += oriented_cylinder_scad(centroid, axis, height, outer_radius, fragments, indent="    ")
    scad += oriented_cylinder_scad(centroid, axis, subtract_height, inner_radius, fragments, indent="    ")
    scad += "}\n"
    return Detection(
        kind="cylindrical_shell_difference",
        confidence=confidence,
        scad=scad,
        notes=[
            "Detected a cylindrical shell or centered cylindrical through-hole.",
            "Output uses difference() with an outer cylinder minus an inner cylinder.",
            "Axis orientation was estimated with PCA.",
        ],
    )


def polyhedron(mesh: "trimesh.Trimesh") -> Detection:
    points_text = ",\n    ".join(vec(vertex) for vertex in mesh.vertices)
    faces_text = ",\n    ".join(vec(face.astype(int)) for face in mesh.faces)
    scad = f"""polyhedron(
    points=[
    {points_text}
    ],
    faces=[
    {faces_text}
    ],
    convexity=10
);
"""
    return Detection(
        kind="polyhedron_fallback",
        confidence=1.0,
        scad=scad,
        notes=[
            "No reliable parametric primitive was detected, so the mesh was embedded as polyhedron().",
            "This preserves geometry as triangles but is not a true editable feature-history rebuild.",
        ],
    )


def detect(mesh: "trimesh.Trimesh", strategy: str, tolerance: float, aggressive: bool = False) -> Detection:
    if strategy == "polyhedron":
        return polyhedron(mesh)

    detector_tolerance = tolerance * (4.0 if aggressive else 1.0)
    candidates: list[Detection] = []
    for detector in (
        box_with_holes,
        cylindrical_shell_difference,
        axis_aligned_box,
        oriented_bounding_box,
        vertical_cylinder,
        pca_oriented_cylinder,
        sphere,
        extruded_profile,
        visual_hull_silhouette,
        silhouette_extrusion,
    ):
        result = detector(mesh, detector_tolerance)
        if result is not None:
            candidates.append(result)

    if candidates:
        boolean_candidates = [
            item
            for item in candidates
            if item.kind
            in {
                "box_with_cylindrical_holes_difference",
                "cylindrical_shell_difference",
            }
        ]
        if boolean_candidates and strategy in {"auto", "primitive"}:
            boolean_candidates.sort(key=lambda item: item.confidence, reverse=True)
            if boolean_candidates[0].confidence >= (0.62 if aggressive else 0.75):
                return boolean_candidates[0]

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        best = candidates[0]
        if strategy == "primitive":
            return best
        if best.confidence >= (0.62 if aggressive else 0.80):
            return best

    fallback = polyhedron(mesh)
    fallback.notes.extend(mesh_diagnostics(mesh, tolerance))
    if aggressive:
        fallback.notes.append("Aggressive mode was enabled, but no candidate cleared the relaxed threshold.")
    else:
        fallback.notes.append("Aggressive mode was not enabled; try rerunning with --aggressive for approximated primitives.")
    return fallback


def write_scad(source: Path, output: Path, detection: Detection) -> None:
    output.write_text(scad_header(source, detection) + detection.scad, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heuristic STL to OpenSCAD parametric converter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input STL file")
    parser.add_argument("-o", "--output", type=Path, help="Output .scad file")
    parser.add_argument(
        "--strategy",
        choices=("auto", "primitive", "polyhedron"),
        default="auto",
        help="Conversion strategy",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Absolute geometric tolerance in model units",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Use looser thresholds and approximation detectors before falling back to polyhedron",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print mesh diagnostics and troubleshooting hints",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source = args.input
    if not source.exists():
        print(f"Input file not found: {source}", file=sys.stderr)
        return 2

    output = args.output or source.with_suffix(".scad")
    mesh = load_mesh(source)
    detection = detect(mesh, args.strategy, args.tolerance, aggressive=args.aggressive)
    write_scad(source, output, detection)

    if not args.quiet:
        print(f"Wrote: {output}")
        print(f"Strategy: {detection.kind}")
        print(f"Confidence: {detection.confidence:.2f}")
        for note in detection.notes:
            print(f"- {note}")
        if args.diagnose and detection.kind != "polyhedron_fallback":
            print("Diagnostics:")
            for note in mesh_diagnostics(mesh, args.tolerance):
                print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
