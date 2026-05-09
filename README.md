# STL-To-OpenScad

Convert STL mesh files to parametric [OpenSCAD](https://openscad.org/) scripts.

Instead of a raw `polyhedron()` dump, the tool analyses the mesh geometry and
emits the simplest parametric primitive that fits:

| Shape detected | OpenSCAD output |
|---|---|
| Axis-aligned box | `cube([w, d, h])` |
| Cylinder | `cylinder(h, r, $fn=…)` |
| Truncated cone (frustum) | `cylinder(h, r1=…, r2=…)` |
| Regular polygon prism (triangle → octagon) | `cylinder(h, r, $fn=n)` |
| Sphere | `sphere(r)` |
| Rotationally symmetric profile (vase, turning) | `rotate_extrude(360) polygon(…)` |
| Cylindrical shell / tube | `difference() { cylinder … cylinder … }` |
| Multiple disconnected bodies | `union() { … }` |
| Anything else | `polyhedron(…)` fallback |

---

## Installation

```bash
pip install trimesh numpy scipy
# dev / testing
pip install -e ".[dev]"
```

## Usage

```bash
# single file
python stl2scad_parametric.py part.stl

# write to file
python stl2scad_parametric.py part.stl -o part.scad

# force a specific strategy
python stl2scad_parametric.py part.stl --strategy oriented_bounding_box_approximation
```

## Running the tests

```bash
pytest -v
```

## How it works

The detector pipeline tries each strategy in order from most-specific to
most-general, returns the first result whose confidence score exceeds the
threshold, and falls back to `polyhedron()` if nothing matches.

See the docstrings in `stl2scad_parametric.py` for details on each detector.
