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
# single file (auto-detects shape)
python stl2scad_parametric.py part.stl

# write to a specific output file
python stl2scad_parametric.py part.stl -o part.scad

# force a specific detector
python stl2scad_parametric.py part.stl --strategy oriented_bounding_box
python stl2scad_parametric.py part.stl --strategy pca_oriented_frustum
python stl2scad_parametric.py part.stl --strategy cylindrical_shell

# always emit a raw polyhedron() (no detection)
python stl2scad_parametric.py part.stl --strategy polyhedron

# loosen thresholds for tricky meshes
python stl2scad_parametric.py part.stl --aggressive

# print mesh diagnostics
python stl2scad_parametric.py part.stl --diagnose
```

### Available `--strategy` values

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Try all detectors, return best match above confidence threshold |
| `primitive` | Return best detector result regardless of confidence |
| `polyhedron` | Always emit raw `polyhedron()`, no detection |
| `axis_aligned_box` | Force axis-aligned box detector |
| `oriented_bounding_box` | Force oriented bounding-box approximation |
| `vertical_cylinder` | Force Z-axis cylinder detector |
| `pca_oriented_cylinder` | Force PCA cylinder detector |
| `pca_oriented_frustum` | Force truncated-cone detector |
| `regular_polygon_prism` | Force polygon prism detector (n=3–8) |
| `sphere` | Force sphere detector |
| `rotational_extrusion` | Force rotate_extrude() detector |
| `extruded_profile` | Force Z-extrusion detector |
| `visual_hull_silhouette` | Force 3-axis visual-hull approximation |
| `silhouette_extrusion` | Force single-axis silhouette extrusion |
| `box_with_holes` | Force box-with-cylindrical-holes detector |
| `cylindrical_shell` | Force cylindrical shell (difference) detector |
| `multi_primitive_union` | Force multi-body union detector |

## Running the tests

```bash
pytest -v
```

## How it works

The detector pipeline tries each strategy in order from most-specific to
most-general, returns the first result whose confidence score exceeds the
threshold, and falls back to `polyhedron()` if nothing matches.

See the docstrings in `stl2scad_parametric.py` for details on each detector.
