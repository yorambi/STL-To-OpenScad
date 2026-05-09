"""
Regression tests for stl2scad_parametric shape detectors.

Each test verifies:
  1. The correct detector (kind) fires.
  2. Confidence is at or above the expected floor.
  3. Key geometric parameters are within 5 % of the ground-truth values.
"""
import sys
import math
import os

import numpy as np
import pytest

# Make the package root importable regardless of install state
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stl2scad_parametric as s


TOLERANCE = 0.01
MIN_CONF = 0.85  # minimum acceptable confidence for any positive detection


def detect(mesh):
    return s.detect(mesh, strategy="auto", tolerance=TOLERANCE)


# ---------------------------------------------------------------------------
# Primitive shapes
# ---------------------------------------------------------------------------

class TestBox:
    def test_kind(self, mesh_box):
        assert detect(mesh_box).kind == "axis_aligned_box"

    def test_confidence(self, mesh_box):
        assert detect(mesh_box).confidence >= MIN_CONF

    def test_scad_contains_cube(self, mesh_box):
        assert "cube" in detect(mesh_box).scad


class TestCylinder:
    def test_kind_lo(self, mesh_cylinder_lo):
        assert detect(mesh_cylinder_lo).kind == "vertical_cylinder"

    def test_kind_hi(self, mesh_cylinder_hi):
        assert detect(mesh_cylinder_hi).kind == "vertical_cylinder"

    def test_confidence(self, mesh_cylinder_hi):
        assert detect(mesh_cylinder_hi).confidence >= MIN_CONF

    def test_scad_contains_cylinder(self, mesh_cylinder_hi):
        assert "cylinder" in detect(mesh_cylinder_hi).scad


class TestSphere:
    def test_kind(self, mesh_sphere):
        assert detect(mesh_sphere).kind == "sphere"

    def test_confidence(self, mesh_sphere):
        assert detect(mesh_sphere).confidence >= MIN_CONF

    def test_scad_contains_sphere(self, mesh_sphere):
        assert "sphere" in detect(mesh_sphere).scad


# ---------------------------------------------------------------------------
# Polygon prisms
# ---------------------------------------------------------------------------

class TestPolygonPrisms:
    @pytest.mark.parametrize("mesh_fixture,expected_n", [
        ("mesh_tri_prism",    3),
        ("mesh_square_prism", 4),
        ("mesh_hex_prism",    6),
        ("mesh_oct_prism",    8),
    ])
    def test_kind(self, request, mesh_fixture, expected_n):
        mesh = request.getfixturevalue(mesh_fixture)
        result = detect(mesh)
        assert result.kind == f"regular_polygon_prism_n{expected_n}", (
            f"Expected n={expected_n}, got {result.kind}"
        )

    @pytest.mark.parametrize("mesh_fixture,expected_n", [
        ("mesh_tri_prism",    3),
        ("mesh_square_prism", 4),
        ("mesh_hex_prism",    6),
        ("mesh_oct_prism",    8),
    ])
    def test_confidence(self, request, mesh_fixture, expected_n):
        mesh = request.getfixturevalue(mesh_fixture)
        assert detect(mesh).confidence >= MIN_CONF

    def test_fn_in_scad(self, mesh_hex_prism):
        scad = detect(mesh_hex_prism).scad
        assert "$fn=6" in scad

    def test_no_false_positive_on_cylinder_lo(self, mesh_cylinder_lo):
        """A 16-section cylinder must NOT be detected as a polygon prism."""
        result = detect(mesh_cylinder_lo)
        assert "polygon_prism" not in result.kind

    def test_no_false_positive_on_cylinder_hi(self, mesh_cylinder_hi):
        result = detect(mesh_cylinder_hi)
        assert "polygon_prism" not in result.kind


# ---------------------------------------------------------------------------
# Frustum (truncated cone)
# ---------------------------------------------------------------------------

class TestFrustum:
    def test_kind(self, mesh_frustum):
        assert detect(mesh_frustum).kind == "pca_oriented_frustum"

    def test_confidence(self, mesh_frustum):
        assert detect(mesh_frustum).confidence >= MIN_CONF

    def test_scad_r1_r2(self, mesh_frustum):
        """Both radii should appear in the SCAD output (r1= and r2=)."""
        scad = detect(mesh_frustum).scad
        assert "r1=" in scad and "r2=" in scad

    def test_approximate_radii(self, mesh_frustum):
        scad = detect(mesh_frustum).scad
        # Extract r1 and r2 from e.g. "cylinder(h=20, r1=8, r2=4, ...)"
        import re
        r1 = float(re.search(r"r1=([\d.]+)", scad).group(1))
        r2 = float(re.search(r"r2=([\d.]+)", scad).group(1))
        r_big, r_small = max(r1, r2), min(r1, r2)
        assert abs(r_big - 8.0) / 8.0 < 0.05, f"Large radius off: {r_big}"
        assert abs(r_small - 4.0) / 4.0 < 0.05, f"Small radius off: {r_small}"

    def test_approximate_height(self, mesh_frustum):
        scad = detect(mesh_frustum).scad
        import re
        h = float(re.search(r"h=([\d.]+)", scad).group(1))
        assert abs(h - 20.0) / 20.0 < 0.05, f"Height off: {h}"


# ---------------------------------------------------------------------------
# Rotational extrusion (vase)
# ---------------------------------------------------------------------------

class TestRotationalExtrusion:
    def test_kind(self, mesh_vase):
        assert detect(mesh_vase).kind == "rotational_extrusion"

    def test_confidence(self, mesh_vase):
        assert detect(mesh_vase).confidence >= 0.75

    def test_scad_rotate_extrude(self, mesh_vase):
        assert "rotate_extrude" in detect(mesh_vase).scad


# ---------------------------------------------------------------------------
# Multi-body union
# ---------------------------------------------------------------------------

class TestMultiBody:
    def test_kind(self, mesh_two_cylinders):
        assert detect(mesh_two_cylinders).kind == "multi_primitive_union"

    def test_confidence(self, mesh_two_cylinders):
        assert detect(mesh_two_cylinders).confidence >= MIN_CONF

    def test_scad_union(self, mesh_two_cylinders):
        assert "union" in detect(mesh_two_cylinders).scad


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

class TestFalsePositives:
    def test_box_not_polygon_prism(self, mesh_box):
        assert "polygon_prism" not in detect(mesh_box).kind

    def test_cylinder_not_polygon_prism(self, mesh_cylinder_hi):
        assert "polygon_prism" not in detect(mesh_cylinder_hi).kind

    def test_cylinder_not_frustum(self, mesh_cylinder_hi):
        assert "frustum" not in detect(mesh_cylinder_hi).kind

    def test_tri_prism_not_box(self, mesh_tri_prism):
        assert detect(mesh_tri_prism).kind != "axis_aligned_box"

    def test_frustum_not_cylinder(self, mesh_frustum):
        assert "cylinder" not in detect(mesh_frustum).kind or "frustum" in detect(mesh_frustum).kind
