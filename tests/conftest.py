"""
Shared mesh fixtures for stl2scad detector tests.
All meshes are generated procedurally — no external STL files needed.
"""
import math
import numpy as np
import pytest
import trimesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frustum(r1: float, r2: float, h: float, sections: int = 32) -> trimesh.Trimesh:
    """Truncated cone (frustum) with r1 at bottom, r2 at top."""
    angles = np.linspace(0, 2 * math.pi, sections, endpoint=False)
    bot = np.column_stack([r1 * np.cos(angles), r1 * np.sin(angles), np.full(sections, -h / 2)])
    top = np.column_stack([r2 * np.cos(angles), r2 * np.sin(angles), np.full(sections, h / 2)])
    bot_c = np.array([[0, 0, -h / 2]])
    top_c = np.array([[0, 0, h / 2]])
    verts = np.vstack([bot, top, bot_c, top_c])
    ic, it = 2 * sections, 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[i, j, sections + j], [i, sections + j, sections + i]]
    for i in range(sections):
        faces.append([ic, (i + 1) % sections, i])
        faces.append([it, sections + i, sections + (i + 1) % sections])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    return m


def _make_vase(sections: int = 64, z_steps: int = 20) -> trimesh.Trimesh:
    """Open-top rotationally symmetric vase."""
    angles = np.linspace(0, 2 * math.pi, sections, endpoint=False)
    zs = np.linspace(0, 20, z_steps)
    rings = []
    for iz, z in enumerate(zs):
        r = 5 + 3 * math.sin(iz / (z_steps - 1) * math.pi)
        rings.append(np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(sections, z)]))
    verts = np.vstack(rings)
    faces = []
    for iz in range(z_steps - 1):
        for i in range(sections):
            j = (i + 1) % sections
            a, b = iz * sections + i, iz * sections + j
            c, d = (iz + 1) * sections + j, (iz + 1) * sections + i
            faces += [[a, b, c], [a, c, d]]
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))


def _make_two_cylinders(r: float = 3.0, h: float = 10.0, gap: float = 10.0) -> trimesh.Trimesh:
    """Two separate cylinders side-by-side (multi-body mesh)."""
    c1 = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    c2 = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    c2.apply_translation([gap, 0, 0])
    return trimesh.util.concatenate([c1, c2])


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_box():
    return trimesh.creation.box(extents=[8, 12, 30])

@pytest.fixture
def mesh_tri_prism():
    return trimesh.creation.cylinder(radius=10, height=25, sections=3)

@pytest.fixture
def mesh_square_prism():
    return trimesh.creation.cylinder(radius=10, height=25, sections=4)

@pytest.fixture
def mesh_hex_prism():
    return trimesh.creation.cylinder(radius=10, height=25, sections=6)

@pytest.fixture
def mesh_oct_prism():
    return trimesh.creation.cylinder(radius=10, height=25, sections=8)

@pytest.fixture
def mesh_cylinder_lo():
    """Low-poly cylinder ($fn=16) — should still resolve as cylinder, not polygon."""
    return trimesh.creation.cylinder(radius=10, height=25, sections=16)

@pytest.fixture
def mesh_cylinder_hi():
    return trimesh.creation.cylinder(radius=10, height=25, sections=64)

@pytest.fixture
def mesh_frustum():
    return _make_frustum(r1=8, r2=4, h=20)

@pytest.fixture
def mesh_sphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=7)

@pytest.fixture
def mesh_vase():
    return _make_vase()

@pytest.fixture
def mesh_two_cylinders():
    return _make_two_cylinders()

@pytest.fixture
def mesh_cylindrical_shell():
    """Hollow tube: outer r=10, inner r=4, h=20."""
    outer = trimesh.creation.cylinder(radius=10, height=20, sections=64)
    inner = trimesh.creation.cylinder(radius=4,  height=21, sections=64)
    return trimesh.boolean.difference(outer, inner) if hasattr(trimesh, "boolean") else outer
